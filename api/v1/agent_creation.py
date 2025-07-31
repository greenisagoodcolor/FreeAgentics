"""API endpoints for agent creation from natural language prompts.

Provides REST endpoints for creating agents using LLM-powered prompt analysis
and system prompt generation. Includes observability, error handling, and
WebSocket support for real-time progress updates.
"""

import logging
import time
from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from agents.creation import AgentFactory
from agents.creation.models import AgentCreationRequest
from core.providers import get_db
from database.models import AgentType

logger = logging.getLogger(__name__)

router = APIRouter()

# Global factory instance - will be initialized properly
agent_factory: Optional[AgentFactory] = None


def get_agent_factory() -> AgentFactory:
    """Get agent factory instance for dependency injection."""
    global agent_factory
    if agent_factory is None:
        agent_factory = AgentFactory()
    return agent_factory


# Request/Response models
class CreateAgentRequest(BaseModel):
    """Request model for creating an agent."""

    prompt: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Natural language description of desired agent",
    )
    agent_name: Optional[str] = Field(
        None, max_length=100, description="Optional custom name for the agent"
    )
    preferred_type: Optional[AgentType] = Field(None, description="Optional preferred agent type")
    enable_advanced_personality: bool = Field(
        True, description="Enable advanced personality generation"
    )
    enable_custom_capabilities: bool = Field(
        True, description="Enable custom capability extraction"
    )

    class Config:
        schema_extra = {
            "example": {
                "prompt": "I need help analyzing market trends and identifying investment opportunities",
                "agent_name": "Market Analyst Pro",
                "preferred_type": "analyst",
                "enable_advanced_personality": True,
                "enable_custom_capabilities": True,
            }
        }


class PreviewAgentRequest(BaseModel):
    """Request model for previewing an agent."""

    prompt: str = Field(
        ..., min_length=10, max_length=1000, description="Natural language description"
    )
    preferred_type: Optional[AgentType] = Field(None, description="Optional preferred agent type")

    class Config:
        schema_extra = {
            "example": {
                "prompt": "Help me critique business proposals and find potential problems",
                "preferred_type": "critic",
            }
        }


class AgentCreationResponse(BaseModel):
    """Response model for agent creation."""

    success: bool
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    agent_type: Optional[str] = None
    processing_time_ms: Optional[int] = None
    error_message: Optional[str] = None

    # Analysis details
    analysis_confidence: Optional[str] = None
    detected_domain: Optional[str] = None
    capabilities: Optional[list[str]] = None

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "agent_id": "agent-123e4567-e89b-12d3-a456-426614174000",
                "agent_name": "Finance Analyst",
                "agent_type": "analyst",
                "processing_time_ms": 1245,
                "analysis_confidence": "high",
                "detected_domain": "finance",
                "capabilities": ["data_analysis", "trend_identification", "reporting"],
            }
        }


class AgentPreviewResponse(BaseModel):
    """Response model for agent preview."""

    agent_name: str
    agent_type: str
    system_prompt: str
    personality_summary: str
    capabilities: list[str]
    confidence: str

    class Config:
        schema_extra = {
            "example": {
                "agent_name": "Business Critic",
                "agent_type": "critic",
                "system_prompt": "You are a Business Critic agent - your role is to identify flaws...",
                "personality_summary": "Highly skeptical and analytical, direct communication style",
                "capabilities": ["critical_analysis", "risk_assessment", "problem_identification"],
                "confidence": "high",
            }
        }


class AgentTypesResponse(BaseModel):
    """Response model for supported agent types."""

    agent_types: list[dict]

    class Config:
        schema_extra = {
            "example": {
                "agent_types": [
                    {
                        "type": "advocate",
                        "description": "Argues for specific positions, builds compelling cases",
                    },
                    {
                        "type": "analyst",
                        "description": "Breaks down complex problems, provides data-driven insights",
                    },
                    {
                        "type": "critic",
                        "description": "Identifies flaws and weaknesses, challenges assumptions",
                    },
                    {
                        "type": "creative",
                        "description": "Generates novel ideas, thinks outside the box",
                    },
                    {
                        "type": "moderator",
                        "description": "Facilitates discussions, maintains balance",
                    },
                ]
            }
        }


@router.post("/create", response_model=AgentCreationResponse)
async def create_agent(
    request: CreateAgentRequest,
    background_tasks: BackgroundTasks,
    factory: AgentFactory = Depends(get_agent_factory),
    db: Session = Depends(get_db),
):
    """Create an agent from a natural language prompt.

    This endpoint analyzes the user's prompt using LLM services to determine
    the optimal agent type, generate a personality profile, and create a
    custom system prompt. The agent is then persisted to the database.

    The process includes:
    1. Prompt analysis to determine agent type and requirements
    2. Personality profile generation based on context
    3. System prompt construction with personality integration
    4. Agent creation and database persistence

    All steps include fallback mechanisms for reliability.
    """

    start_time = time.time()

    try:
        logger.info(f"Creating agent from prompt: {request.prompt[:100]}...")

        # Convert API request to internal request format
        creation_request = AgentCreationRequest(
            prompt=request.prompt,
            agent_name=request.agent_name,
            preferred_type=request.preferred_type,
            enable_advanced_personality=request.enable_advanced_personality,
            enable_custom_capabilities=request.enable_custom_capabilities,
            preview_only=False,
        )

        # Create the agent
        result = await factory.create_agent(creation_request)

        if not result.success:
            raise HTTPException(
                status_code=500, detail=f"Agent creation failed: {result.error_message}"
            )

        # Build response
        response = AgentCreationResponse(
            success=True,
            agent_id=str(result.agent.id) if result.agent else None,
            agent_name=result.agent.name if result.agent else None,
            agent_type=result.agent.agent_type.value if result.agent else None,
            processing_time_ms=result.processing_time_ms,
            analysis_confidence=result.analysis_result.confidence.value
            if result.analysis_result
            else None,
            detected_domain=result.analysis_result.domain if result.analysis_result else None,
            capabilities=result.analysis_result.capabilities if result.analysis_result else None,
        )

        # Log success metrics
        processing_time = int((time.time() - start_time) * 1000)
        logger.info(f"Agent created successfully in {processing_time}ms: {response.agent_id}")

        return response

    except HTTPException:
        raise
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        logger.error(f"Agent creation failed after {processing_time}ms: {e}")

        return AgentCreationResponse(
            success=False, error_message=str(e), processing_time_ms=processing_time
        )


@router.post("/preview", response_model=AgentPreviewResponse)
async def preview_agent(
    request: PreviewAgentRequest, factory: AgentFactory = Depends(get_agent_factory)
):
    """Preview what an agent would look like without creating it.

    This endpoint performs the same analysis and generation steps as
    agent creation but doesn't persist the agent to the database.
    Useful for users to preview and refine their prompts before
    committing to agent creation.
    """

    try:
        logger.info(f"Previewing agent from prompt: {request.prompt[:100]}...")

        # Create preview request
        creation_request = AgentCreationRequest(
            prompt=request.prompt, preferred_type=request.preferred_type, preview_only=True
        )

        # Generate preview
        result = await factory.create_agent(creation_request)

        if not result.success or not result.specification:
            raise HTTPException(
                status_code=500, detail=f"Preview generation failed: {result.error_message}"
            )

        spec = result.specification

        # Generate personality summary
        personality_traits = []
        if spec.personality.assertiveness > 0.7:
            personality_traits.append("highly assertive")
        elif spec.personality.assertiveness < 0.3:
            personality_traits.append("gentle and collaborative")

        if spec.personality.analytical_depth > 0.7:
            personality_traits.append("very analytical")
        elif spec.personality.creativity > 0.7:
            personality_traits.append("highly creative")

        if spec.personality.skepticism > 0.7:
            personality_traits.append("very skeptical")
        elif spec.personality.empathy > 0.7:
            personality_traits.append("empathetic")

        personality_summary = (
            ", ".join(personality_traits) if personality_traits else "balanced personality"
        )

        response = AgentPreviewResponse(
            agent_name=spec.name,
            agent_type=spec.agent_type.value,
            system_prompt=spec.system_prompt[:200] + "..."
            if len(spec.system_prompt) > 200
            else spec.system_prompt,
            personality_summary=personality_summary,
            capabilities=spec.capabilities,
            confidence=result.analysis_result.confidence.value
            if result.analysis_result
            else "medium",
        )

        logger.info(f"Agent preview generated: {spec.agent_type.value} - {spec.name}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent preview failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")


@router.get("/types", response_model=AgentTypesResponse)
async def get_agent_types(factory: AgentFactory = Depends(get_agent_factory)):
    """Get list of supported agent types with descriptions.

    Returns information about all available agent types that can be
    created through natural language prompts, including their
    capabilities and typical use cases.
    """

    try:
        agent_types = await factory.get_supported_agent_types()

        type_descriptions = {
            AgentType.ADVOCATE: "Argues for specific positions, builds compelling cases, persuades others",
            AgentType.ANALYST: "Breaks down complex problems, provides data-driven insights, systematic thinking",
            AgentType.CRITIC: "Identifies flaws and weaknesses, challenges assumptions, points out problems",
            AgentType.CREATIVE: "Generates novel ideas, thinks outside the box, innovative solutions",
            AgentType.MODERATOR: "Facilitates discussions, maintains balance, ensures fair participation",
        }

        response = AgentTypesResponse(
            agent_types=[
                {
                    "type": agent_type.value,
                    "description": type_descriptions.get(agent_type, "Agent type description"),
                }
                for agent_type in agent_types
            ]
        )

        return response

    except Exception as e:
        logger.error(f"Failed to get agent types: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve agent types")


@router.get("/metrics")
async def get_creation_metrics(factory: AgentFactory = Depends(get_agent_factory)):
    """Get metrics for agent creation system monitoring.

    Returns performance and reliability metrics for the agent creation
    system, including success rates, average processing times, and
    fallback usage statistics.
    """

    try:
        metrics = factory.get_metrics()

        # Add system health indicators
        metrics["system_health"] = "healthy" if metrics["success_rate"] > 0.9 else "degraded"
        metrics["fallback_health"] = "good" if metrics["fallback_rate"] < 0.2 else "high_usage"

        return metrics

    except Exception as e:
        logger.error(f"Failed to get creation metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


# WebSocket endpoint for real-time agent creation progress
@router.websocket("/create/ws")
async def create_agent_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time agent creation with progress updates.

    Accepts agent creation requests over WebSocket and provides real-time
    progress updates during the creation process. Useful for long-running
    agent creation tasks that involve multiple LLM calls.

    Message format:
    - Incoming: {"prompt": "...", "agent_name": "...", "preferred_type": "..."}
    - Outgoing: {"status": "progress|success|error", "message": "...", "data": {...}}
    """

    await websocket.accept()
    factory = get_agent_factory()

    try:
        while True:
            # Receive creation request
            data = await websocket.receive_json()

            try:
                # Validate request
                request = CreateAgentRequest(**data)

                # Send progress update
                await websocket.send_json(
                    {
                        "status": "progress",
                        "message": "Analyzing prompt...",
                        "step": 1,
                        "total_steps": 4,
                    }
                )

                # Convert to internal request
                creation_request = AgentCreationRequest(
                    prompt=request.prompt,
                    agent_name=request.agent_name,
                    preferred_type=request.preferred_type,
                    enable_advanced_personality=request.enable_advanced_personality,
                    enable_custom_capabilities=request.enable_custom_capabilities,
                    preview_only=False,
                )

                # Send progress update
                await websocket.send_json(
                    {
                        "status": "progress",
                        "message": "Generating personality profile...",
                        "step": 2,
                        "total_steps": 4,
                    }
                )

                # Create agent
                result = await factory.create_agent(creation_request)

                if result.success:
                    await websocket.send_json(
                        {
                            "status": "success",
                            "message": "Agent created successfully!",
                            "data": {
                                "agent_id": str(result.agent.id) if result.agent else None,
                                "agent_name": result.agent.name if result.agent else None,
                                "agent_type": result.agent.agent_type.value
                                if result.agent
                                else None,
                                "processing_time_ms": result.processing_time_ms,
                            },
                        }
                    )
                else:
                    await websocket.send_json(
                        {
                            "status": "error",
                            "message": f"Agent creation failed: {result.error_message}",
                        }
                    )

            except Exception as e:
                await websocket.send_json(
                    {"status": "error", "message": f"Request failed: {str(e)}"}
                )

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"status": "error", "message": f"Connection error: {str(e)}"})
        except Exception:
            pass  # Connection already closed

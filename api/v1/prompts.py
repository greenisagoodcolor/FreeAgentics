"""Prompt processing API endpoint for FreeAgentics platform.

This endpoint handles natural language prompts and orchestrates the
prompt → agent → knowledge graph pipeline.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

# These will be injected via dependency injection
from agents.agent_manager import AgentManager
from agents.pymdp_adapter import PyMDPCompatibilityAdapter

# Import WebSocket broadcasting functions
from api.v1.websocket import broadcast_agent_event, broadcast_system_event
from auth.security_implementation import (
    Permission,
    TokenData,
    get_current_user,
    require_permission,
)
from database.session import get_db
from inference.active.gmn_parser import GMNParser
from knowledge_graph.graph_engine import KnowledgeGraph
from services.agent_factory import AgentFactory
from services.belief_kg_bridge import BeliefKGBridge
from services.gmn_generator import GMNGenerator
from services.iterative_controller import IterativeController
from services.prompt_processor import PromptProcessor

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response models
class PromptRequest(BaseModel):
    """Request model for prompt processing."""

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language prompt for agent creation or modification",
    )
    conversation_id: Optional[str] = Field(
        None, description="Optional conversation ID for context continuation"
    )
    iteration_count: Optional[int] = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of GMN refinement iterations",
    )

    @validator('prompt')
    def validate_prompt(cls, v):
        """Ensure prompt is not just whitespace."""
        if not v.strip():
            raise ValueError("Prompt cannot be empty or just whitespace")
        return v.strip()


class PromptResponse(BaseModel):
    """Response model for prompt processing."""

    agent_id: str = Field(..., description="ID of created/modified agent")
    gmn_specification: str = Field(
        ..., description="Generated GMN specification"
    )
    knowledge_graph_updates: List[Dict[str, Any]] = Field(
        ..., description="List of knowledge graph updates applied"
    )
    next_suggestions: List[str] = Field(
        ..., description="Suggested next actions or refinements"
    )
    status: str = Field(
        ..., description="Processing status: success, partial_success, failed"
    )
    warnings: Optional[List[str]] = Field(
        None, description="Any warnings during processing"
    )
    processing_time_ms: Optional[float] = Field(
        None, description="Processing time in milliseconds"
    )
    iteration_context: Optional[Dict[str, Any]] = Field(
        None, description="Context about the iterative conversation"
    )


# Dependency injection
_prompt_processor: Optional[PromptProcessor] = None


async def websocket_pipeline_callback(event_type: str, data: Dict[str, Any]):
    """Callback for WebSocket updates during pipeline processing."""
    # Broadcast pipeline events to WebSocket subscribers
    await broadcast_system_event(f"pipeline:{event_type}", data)

    # Also broadcast specific agent events
    if event_type == "agent_created" and "agent_id" in data:
        await broadcast_agent_event(
            data["agent_id"],
            "created",
            {
                "prompt_id": data.get("prompt_id"),
                "agent_type": data.get("agent_type"),
            },
        )
    elif event_type == "knowledge_graph_updated" and "prompt_id" in data:
        await broadcast_system_event(
            "knowledge_graph:updated",
            {
                "prompt_id": data["prompt_id"],
                "updates_count": data.get("updates_count", 0),
                "nodes_added": data.get("nodes_added", 0),
            },
        )


def get_prompt_processor() -> PromptProcessor:
    """Get or create prompt processor instance."""
    global _prompt_processor

    if _prompt_processor is None:
        # Initialize with real implementations
        gmn_generator = GMNGenerator()  # Uses MockLLMProvider by default
        gmn_parser = GMNParser()
        agent_factory = AgentFactory()
        agent_manager = AgentManager()
        knowledge_graph = KnowledgeGraph()
        belief_kg_bridge = BeliefKGBridge()
        pymdp_adapter = PyMDPCompatibilityAdapter()

        # Create iterative controller
        iterative_controller = IterativeController(
            knowledge_graph=knowledge_graph,
            belief_kg_bridge=belief_kg_bridge,
            pymdp_adapter=pymdp_adapter,
        )

        _prompt_processor = PromptProcessor(
            gmn_generator=gmn_generator,
            gmn_parser=gmn_parser,
            agent_factory=agent_factory,
            agent_manager=agent_manager,
            knowledge_graph=knowledge_graph,
            belief_kg_bridge=belief_kg_bridge,
            pymdp_adapter=pymdp_adapter,
            iterative_controller=iterative_controller,
            websocket_callback=websocket_pipeline_callback,
        )

    return _prompt_processor


@router.post("/prompts", response_model=PromptResponse)
async def process_prompt(
    request: PromptRequest,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    prompt_processor: PromptProcessor = Depends(get_prompt_processor),
    _: None = Depends(require_permission(Permission.CREATE_AGENT)),
) -> PromptResponse:
    """Process a natural language prompt to create or modify an agent.

    This endpoint orchestrates the full prompt → agent → knowledge graph pipeline:
    1. Converts natural language to GMN specification via LLM
    2. Validates and parses GMN into PyMDP model
    3. Creates active inference agent from model
    4. Updates knowledge graph with agent beliefs
    5. Returns suggestions for next actions

    Args:
        request: The prompt request containing text and optional parameters
        current_user: Authenticated user from JWT token
        db: Database session
        prompt_processor: Injected prompt processing service

    Returns:
        PromptResponse with agent ID, GMN spec, KG updates, and suggestions

    Raises:
        400: If GMN validation fails
        401: If user is not authenticated
        403: If user lacks CREATE_AGENT permission
        422: If request validation fails
        500: If agent creation fails
    """
    import time

    start_time = time.time()

    logger.info(
        f"Processing prompt for user {current_user.username}: {request.prompt[:100]}..."
    )

    try:
        # Process the prompt
        result = await prompt_processor.process_prompt(
            prompt_text=request.prompt,
            user_id=current_user.username,
            db=db,
            conversation_id=request.conversation_id,
            iteration_count=request.iteration_count,
        )

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        # Add processing time to result
        result["processing_time_ms"] = processing_time

        # Log success
        logger.info(
            f"Successfully created agent {result['agent_id']} in {processing_time:.2f}ms"
        )

        return PromptResponse(**result)

    except ValueError as e:
        # GMN validation error
        logger.error(f"GMN validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )

    except RuntimeError as e:
        # Agent creation error
        logger.error(f"Agent creation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

    except Exception as e:
        # Unexpected error
        logger.error(
            f"Unexpected error processing prompt: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )


@router.get("/prompts/templates")
async def get_prompt_templates(
    category: Optional[str] = None,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Get available prompt templates.

    Returns a list of pre-defined prompt templates that can be used
    as starting points for agent creation.

    Args:
        category: Optional filter by template category
        current_user: Authenticated user
        db: Database session

    Returns:
        List of prompt templates with examples
    """
    # TODO: Implement template retrieval from database
    templates = [
        {
            "id": "explorer-basic",
            "name": "Basic Explorer Agent",
            "category": "explorer",
            "description": "Simple grid world exploration agent",
            "example_prompt": "Create an explorer agent for a 5x5 grid world",
            "suggested_parameters": {
                "grid_size": [5, 5],
                "planning_horizon": 3,
            },
        },
        {
            "id": "trader-market",
            "name": "Market Trader Agent",
            "category": "trader",
            "description": "Agent for trading in a market environment",
            "example_prompt": "Create a trader agent that can buy and sell resources",
            "suggested_parameters": {
                "resource_types": ["gold", "silver", "copper"],
                "initial_capital": 1000,
            },
        },
    ]

    if category:
        templates = [t for t in templates if t["category"] == category]

    return templates


@router.get("/prompts/suggestions")
async def get_contextual_suggestions(
    agent_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Get contextual suggestions for an existing agent.

    Analyzes the current state of an agent and provides suggestions
    for improvements or next actions.

    Args:
        agent_id: ID of the agent to analyze
        current_user: Authenticated user
        db: Database session

    Returns:
        Dictionary containing suggestions and agent analysis
    """
    # TODO: Implement actual agent analysis
    return {
        "agent_id": agent_id,
        "current_state": {
            "belief_entropy": 0.75,
            "exploration_coverage": 0.3,
            "goal_progress": 0.5,
        },
        "suggestions": [
            "Add curiosity-driven exploration to reduce uncertainty",
            "Define clearer goal states for directed behavior",
            "Consider forming a coalition for complex tasks",
        ],
        "recommended_prompts": [
            "Make the agent more curious about unexplored areas",
            "Add a specific goal location to the agent's preferences",
            "Create a coordinator agent to work with this explorer",
        ],
    }

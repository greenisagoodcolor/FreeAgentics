"""Agent API endpoints for FreeAgentics platform."""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Security imports
from auth.security_implementation import Permission, TokenData, get_current_user, require_permission

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models for request/response
class AgentConfig(BaseModel):
    """Configuration for creating a new agent."""

    name: str = Field(..., min_length=1, max_length=100)
    template: str = Field(..., description="Agent template ID")
    parameters: Optional[dict] = Field(default_factory=dict)
    gmn_spec: Optional[str] = Field(
        None, description="GMN specification for Active Inference model"
    )
    use_pymdp: Optional[bool] = Field(True, description="Whether to use PyMDP for Active Inference")
    planning_horizon: Optional[int] = Field(3, description="Planning horizon for Active Inference")


class Agent(BaseModel):
    """Agent model representing an active inference agent."""

    id: str
    name: str
    template: str
    status: str = Field(..., description="pending, active, paused, stopped")
    created_at: datetime
    last_active: Optional[datetime] = None
    inference_count: int = 0
    parameters: dict = Field(default_factory=dict)


class AgentMetrics(BaseModel):
    """Metrics for an individual agent."""

    agent_id: str
    total_inferences: int
    avg_response_time: float
    memory_usage: float
    last_update: datetime


from fastapi import Depends

# Database imports - NO IN-MEMORY STORAGE
from sqlalchemy.orm import Session

from database.models import Agent as AgentModel
from database.models import AgentStatus as DBAgentStatus
from database.session import get_db

# Import agent manager
try:
    from agents.agent_manager import AgentManager

    agent_manager = AgentManager()
    # Create default world
    agent_manager.create_world(size=20)
    AGENT_MANAGER_AVAILABLE = True
except ImportError:
    logger.warning("Agent manager not available")
    agent_manager = None
    AGENT_MANAGER_AVAILABLE = False


@router.post("/agents", response_model=Agent, status_code=201)
@require_permission(Permission.CREATE_AGENT)
async def create_agent(
    config: AgentConfig,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Agent:
    """Create a new active inference agent with optional GMN specification.

    This creates a real database record with support for:
    - GMN (Generalized Notation Notation) model specifications
    - PyMDP Active Inference configuration
    - Planning horizon settings
    """
    # Prepare agent parameters with GMN and PyMDP settings
    agent_parameters = config.parameters.copy() if config.parameters else {}

    # Add Active Inference configuration
    if config.use_pymdp is not None:
        agent_parameters["use_pymdp"] = config.use_pymdp
    if config.planning_horizon is not None:
        agent_parameters["planning_horizon"] = config.planning_horizon

    # Create database model
    db_agent = AgentModel(
        name=config.name,
        template=config.template,
        status=DBAgentStatus.PENDING,
        parameters=agent_parameters,
        gmn_spec=config.gmn_spec,  # Store GMN specification in database
    )

    # Save to database
    db.add(db_agent)
    db.commit()
    db.refresh(db_agent)

    logger.info(f"Created agent in DB: {db_agent.id} with template: {config.template}")

    # Convert to API model
    return Agent(
        id=str(db_agent.id),
        name=db_agent.name,
        template=db_agent.template,
        status=db_agent.status.value,
        created_at=db_agent.created_at,
        parameters=db_agent.parameters,
        inference_count=db_agent.inference_count,
    )


@router.get("/agents", response_model=List[Agent])
@require_permission(Permission.VIEW_AGENTS)
async def list_agents(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000),
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[Agent]:
    """List all agents with optional filtering.

    Fetches from real database, not memory.
    """
    query = db.query(AgentModel)

    if status:
        try:
            status_enum = DBAgentStatus(status)
            query = query.filter(AgentModel.status == status_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    db_agents = query.limit(limit).all()

    # Convert to API models
    return [
        Agent(
            id=str(agent.id),
            name=agent.name,
            template=agent.template,
            status=agent.status.value,
            created_at=agent.created_at,
            last_active=agent.last_active,
            inference_count=agent.inference_count,
            parameters=agent.parameters,
        )
        for agent in db_agents
    ]


@router.get("/agents/{agent_id}", response_model=Agent)
@require_permission(Permission.VIEW_AGENTS)
async def get_agent(
    agent_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Agent:
    """Get details of a specific agent.

    Fetches from database by UUID.
    """
    try:
        # Parse UUID
        from uuid import UUID

        agent_uuid = UUID(agent_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid agent ID format: {agent_id}")

    db_agent = db.query(AgentModel).filter(AgentModel.id == agent_uuid).first()
    if not db_agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    return Agent(
        id=str(db_agent.id),
        name=db_agent.name,
        template=db_agent.template,
        status=db_agent.status.value,
        created_at=db_agent.created_at,
        last_active=db_agent.last_active,
        inference_count=db_agent.inference_count,
        parameters=db_agent.parameters,
    )


@router.patch("/agents/{agent_id}/status")
@require_permission(Permission.MODIFY_AGENT)
async def update_agent_status(
    agent_id: str,
    status: str,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Update agent status (start, stop, pause).

    Updates in database and optionally in agent manager.
    """
    try:
        from uuid import UUID

        agent_uuid = UUID(agent_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid agent ID format: {agent_id}")

    db_agent = db.query(AgentModel).filter(AgentModel.id == agent_uuid).first()
    if not db_agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    # Validate status
    try:
        status_enum = DBAgentStatus(status)
    except ValueError:
        valid_statuses = [s.value for s in DBAgentStatus]
        raise HTTPException(
            status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}"
        )

    # Update database
    db_agent.status = status_enum
    db_agent.last_active = datetime.now()
    db.commit()

    # Update in agent manager if available
    if AGENT_MANAGER_AVAILABLE and agent_manager:
        if status == "active":
            # TODO: Create actual agent instance from DB record
            pass
        elif status == "stopped":
            agent_manager.remove_agent(str(agent_uuid))

    logger.info(f"Updated agent {agent_id} status to: {status}")

    return {"agent_id": agent_id, "status": status}


@router.delete("/agents/{agent_id}")
@require_permission(Permission.DELETE_AGENT)
async def delete_agent(
    agent_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Delete an agent.

    Removes from database and agent manager.
    """
    try:
        from uuid import UUID

        agent_uuid = UUID(agent_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid agent ID format: {agent_id}")

    db_agent = db.query(AgentModel).filter(AgentModel.id == agent_uuid).first()
    if not db_agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    # Remove from agent manager if present
    if AGENT_MANAGER_AVAILABLE and agent_manager:
        agent_manager.remove_agent(str(agent_uuid))

    # Delete from database
    db.delete(db_agent)
    db.commit()

    logger.info(f"Deleted agent: {agent_id}")

    return {"message": f"Agent {agent_id} deleted successfully"}


@router.get("/agents/{agent_id}/metrics", response_model=AgentMetrics)
@require_permission(Permission.VIEW_METRICS)
async def get_agent_metrics(
    agent_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AgentMetrics:
    """Get performance metrics for a specific agent.

    Fetches from database and computes real metrics.
    """
    try:
        from uuid import UUID

        agent_uuid = UUID(agent_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid agent ID format: {agent_id}")

    db_agent = db.query(AgentModel).filter(AgentModel.id == agent_uuid).first()
    if not db_agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    # Get real metrics from database
    metrics_data = db_agent.metrics or {}

    metrics = AgentMetrics(
        agent_id=agent_id,
        total_inferences=db_agent.inference_count,
        avg_response_time=metrics_data.get("avg_response_time", 0.0),
        memory_usage=metrics_data.get("memory_usage", 0.0),
        last_update=db_agent.updated_at,
    )

    return metrics


# GMN-specific endpoints
class GMNAgentRequest(BaseModel):
    """Request model for creating agents from GMN specifications."""

    name: str = Field(..., min_length=1, max_length=100)
    gmn_spec: str = Field(..., description="GMN specification string or JSON")
    template: str = Field("gmn_agent", description="Agent template for GMN-based agents")
    planning_horizon: Optional[int] = Field(3, description="Planning horizon for Active Inference")


@router.post("/agents/from-gmn", response_model=Agent, status_code=201)
@require_permission(Permission.CREATE_AGENT)
async def create_agent_from_gmn(
    request: GMNAgentRequest,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Agent:
    """Create an Active Inference agent from a GMN specification.

    This endpoint:
    1. Parses the GMN specification
    2. Validates the GMN model structure
    3. Creates a PyMDP-compatible agent
    4. Stores the agent in the database

    GMN (Generalized Notation Notation) allows you to specify:
    - State spaces and observations
    - Transition and observation models
    - Preferences and initial beliefs
    - LLM integration points
    """
    try:
        # Import GMN parser
        from inference.active.gmn_parser import GMNParser

        # Parse and validate GMN specification
        parser = GMNParser()
        gmn_graph = parser.parse(request.gmn_spec)
        pymdp_model = parser.to_pymdp_model(gmn_graph)

        # Create agent with parsed GMN model
        agent_parameters = {
            "use_pymdp": True,
            "planning_horizon": request.planning_horizon,
            "gmn_parsed": True,
            "model_dimensions": {
                "num_states": pymdp_model.get("num_states", []),
                "num_obs": pymdp_model.get("num_obs", []),
                "num_actions": pymdp_model.get("num_actions", []),
            },
        }

        # Create database record
        db_agent = AgentModel(
            name=request.name,
            template=request.template,
            status=DBAgentStatus.PENDING,
            parameters=agent_parameters,
            gmn_spec=request.gmn_spec,  # Store original GMN spec
        )

        # Save to database
        db.add(db_agent)
        db.commit()
        db.refresh(db_agent)

        logger.info(f"Created GMN agent in DB: {db_agent.id} with {len(gmn_graph.nodes)} GMN nodes")

        # Return API model
        return Agent(
            id=str(db_agent.id),
            name=db_agent.name,
            template=db_agent.template,
            status=db_agent.status.value,
            created_at=db_agent.created_at,
            parameters=db_agent.parameters,
            inference_count=db_agent.inference_count,
        )

    except Exception as e:
        logger.error(f"Failed to create agent from GMN: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid GMN specification: {str(e)}")


@router.get("/agents/{agent_id}/gmn", response_model=dict)
@require_permission(Permission.VIEW_AGENTS)
async def get_agent_gmn_spec(
    agent_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Get the GMN specification for an agent."""
    try:
        from uuid import UUID

        agent_uuid = UUID(agent_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid agent ID format: {agent_id}")

    db_agent = db.query(AgentModel).filter(AgentModel.id == agent_uuid).first()
    if not db_agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    if not db_agent.gmn_spec:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} has no GMN specification")

    return {
        "agent_id": agent_id,
        "gmn_spec": db_agent.gmn_spec,
        "parsed": db_agent.parameters.get("gmn_parsed", False),
        "model_dimensions": db_agent.parameters.get("model_dimensions", {}),
    }


@router.put("/agents/{agent_id}/gmn")
@require_permission(Permission.MODIFY_AGENT)
async def update_agent_gmn_spec(
    agent_id: str,
    gmn_spec: str,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    """Update the GMN specification for an existing agent."""
    try:
        from uuid import UUID

        agent_uuid = UUID(agent_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid agent ID format: {agent_id}")

    db_agent = db.query(AgentModel).filter(AgentModel.id == agent_uuid).first()
    if not db_agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    try:
        # Validate new GMN specification
        from inference.active.gmn_parser import GMNParser

        parser = GMNParser()
        gmn_graph = parser.parse(gmn_spec)
        pymdp_model = parser.to_pymdp_model(gmn_graph)

        # Update database record
        db_agent.gmn_spec = gmn_spec
        db_agent.parameters = db_agent.parameters or {}
        db_agent.parameters.update(
            {
                "gmn_parsed": True,
                "model_dimensions": {
                    "num_states": pymdp_model.get("num_states", []),
                    "num_obs": pymdp_model.get("num_obs", []),
                    "num_actions": pymdp_model.get("num_actions", []),
                },
            }
        )

        db.commit()

        logger.info(f"Updated GMN spec for agent {agent_id}")

        return {
            "message": f"GMN specification updated for agent {agent_id}",
            "model_dimensions": db_agent.parameters["model_dimensions"],
        }

    except Exception as e:
        logger.error(f"Failed to update GMN spec: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid GMN specification: {str(e)}")


@router.get("/gmn/examples")
@require_permission(Permission.VIEW_AGENTS)
async def get_gmn_examples(current_user: TokenData = Depends(get_current_user)) -> dict:
    """Get example GMN specifications for different agent types."""
    from inference.active.gmn_parser import EXAMPLE_GMN_SPEC

    examples = {
        "grid_explorer": EXAMPLE_GMN_SPEC,
        "resource_collector": """
[nodes]
position: state {num_states: 25}
obs_position: observation {num_observations: 6}
move: action {num_actions: 6}
position_belief: belief
exploration_pref: preference {preferred_observation: 1}
position_likelihood: likelihood
position_transition: transition
collect_action: action {num_actions: 2}
resource_detector: llm_query {trigger_condition: "on_observation", prompt_template: "Analyze resource availability at {position}"}

[edges]
position -> position_likelihood: depends_on
position_likelihood -> obs_position: generates
position -> position_transition: depends_on
move -> position_transition: depends_on
exploration_pref -> obs_position: depends_on
position_belief -> position: depends_on
obs_position -> resource_detector: queries
resource_detector -> collect_action: updates
""",
        "coalition_coordinator": """
[nodes]
agent_states: state {num_states: 16}
coalition_obs: observation {num_observations: 8}
coordination_action: action {num_actions: 4}
coalition_belief: belief
cooperation_pref: preference {preferred_observation: 2}
coalition_likelihood: likelihood
coalition_transition: transition
strategy_planner: llm_query {trigger_condition: "on_coalition_change", prompt_template: "Plan coordination strategy for agents: {agent_list}"}

[edges]
agent_states -> coalition_likelihood: depends_on
coalition_likelihood -> coalition_obs: generates
agent_states -> coalition_transition: depends_on
coordination_action -> coalition_transition: depends_on
cooperation_pref -> coalition_obs: depends_on
coalition_belief -> agent_states: depends_on
coalition_obs -> strategy_planner: queries
strategy_planner -> coordination_action: updates
""",
    }

    return {
        "description": "Example GMN specifications for different agent types",
        "examples": examples,
        "documentation": "GMN format: [nodes] section defines components, [edges] section defines relationships",
    }


# Agent template endpoints
@router.get("/templates")
@require_permission(Permission.VIEW_AGENTS)
async def list_agent_templates(current_user: TokenData = Depends(get_current_user)) -> List[dict]:
    """List available agent templates."""
    templates = [
        {
            "id": "basic-explorer",
            "name": "Basic Explorer",
            "description": "Simple agent that explores the environment using active inference",
            "type": "explorer",
            "complexity": "simple",
            "parameters": {"exploration_rate": 0.3, "planning_horizon": 5},
        },
        {
            "id": "goal-optimizer",
            "name": "Goal Optimizer",
            "description": "Agent focused on optimizing specific objectives",
            "type": "optimizer",
            "complexity": "moderate",
            "parameters": {"optimization_target": "efficiency", "learning_rate": 0.01},
        },
        {
            "id": "pattern-predictor",
            "name": "Pattern Predictor",
            "description": "Advanced agent that learns and predicts environmental patterns",
            "type": "predictor",
            "complexity": "complex",
            "parameters": {"prediction_window": 10, "model_complexity": "high"},
        },
    ]

    return templates

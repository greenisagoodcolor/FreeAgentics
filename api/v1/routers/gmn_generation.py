"""FastAPI router for GMN generation from natural language.

This router provides endpoints for converting natural language descriptions
into Generative Model Notation (GMN) specifications for active inference agents.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session

from database.gmn_versioned_repository import GMNVersionedRepository
from database.session import get_db
from services.gmn_generator import GMNGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gmn", tags=["GMN Generation"])


# Request/Response Models
class GMNGenerationRequest(BaseModel):
    """Request model for GMN generation."""

    prompt: str = Field(
        ..., min_length=10, max_length=2000, description="Natural language description"
    )
    agent_type: str = Field(
        default="general", description="Type of agent (explorer, trader, coordinator, general)"
    )
    agent_id: Optional[str] = Field(None, description="Agent ID for versioning")
    constraints: Optional[Dict[str, Any]] = Field(
        None, description="Optional constraints on GMN structure"
    )
    name: Optional[str] = Field(None, max_length=100, description="Name for the generated GMN")

    @validator("agent_type")
    def validate_agent_type(cls, v):
        """Validate agent type is supported."""
        valid_types = {"explorer", "trader", "coordinator", "general"}
        if v not in valid_types:
            raise ValueError(f"Agent type must be one of: {', '.join(valid_types)}")
        return v

    @validator("prompt")
    def validate_prompt_content(cls, v):
        """Validate prompt has meaningful content."""
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v.strip()


class GMNGenerationResponse(BaseModel):
    """Response model for GMN generation."""

    gmn_specification: str = Field(..., description="Generated GMN specification")
    specification_id: str = Field(..., description="Unique ID for the specification")
    agent_id: str = Field(..., description="Agent ID this GMN belongs to")
    version_number: int = Field(..., description="Version number of this specification")
    validation_status: str = Field(..., description="Validation status (valid, invalid, warning)")
    validation_errors: List[str] = Field(
        default_factory=list, description="Validation errors if any"
    )
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class GMNValidationRequest(BaseModel):
    """Request model for GMN validation."""

    gmn_specification: str = Field(..., min_length=1, description="GMN specification to validate")


class GMNValidationResponse(BaseModel):
    """Response model for GMN validation."""

    is_valid: bool = Field(..., description="Whether the GMN is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")


class GMNRefinementRequest(BaseModel):
    """Request model for GMN refinement."""

    gmn_specification: str = Field(..., min_length=1, description="Current GMN specification")
    feedback: str = Field(..., min_length=5, description="Feedback or errors to address")


# Dependency injection
def get_gmn_generator() -> GMNGenerator:
    """Get GMN generator instance."""
    return GMNGenerator()


def get_gmn_repository(db: Session = Depends(get_db)) -> GMNVersionedRepository:
    """Get GMN repository instance."""
    return GMNVersionedRepository(db)


@router.post("/generate", response_model=GMNGenerationResponse, status_code=status.HTTP_201_CREATED)
async def generate_gmn(
    request: GMNGenerationRequest,
    generator: GMNGenerator = Depends(get_gmn_generator),
    repository: GMNVersionedRepository = Depends(get_gmn_repository),
) -> GMNGenerationResponse:
    """Generate GMN specification from natural language prompt.

    This endpoint converts a natural language description into a structured
    GMN specification suitable for active inference agents.
    """
    try:
        # Generate agent ID if not provided
        agent_uuid = uuid.UUID(request.agent_id) if request.agent_id else uuid.uuid4()

        # Generate the GMN specification
        logger.info(f"Generating GMN for agent {agent_uuid} with prompt: {request.prompt[:100]}...")

        gmn_spec = await generator.prompt_to_gmn(
            prompt=request.prompt, agent_type=request.agent_type, constraints=request.constraints
        )

        # Validate the generated GMN
        is_valid, validation_errors = await generator.validate_gmn(gmn_spec)

        # Get improvement suggestions
        suggestions = await generator.suggest_improvements(gmn_spec)

        # Parse GMN for storage
        try:
            # Simple parsing for node/edge counts
            parsed_data = _parse_gmn_basic(gmn_spec)
        except Exception as e:
            logger.warning(f"Failed to parse GMN for storage metrics: {e}")
            parsed_data = {}

        # Store the specification with versioning
        gmn_name = request.name or f"{request.agent_type}_agent_{agent_uuid}"

        stored_spec = repository.create_gmn_specification_versioned(
            agent_id=agent_uuid,
            specification=gmn_spec,
            name=gmn_name,
            parsed_data=parsed_data,
            version_metadata={
                "agent_type": request.agent_type,
                "original_prompt": request.prompt,
                "constraints": request.constraints or {},
                "generation_timestamp": "utc_now",  # Will be set by repository
            },
        )

        # Determine validation status
        validation_status = "valid" if is_valid else "invalid"
        if is_valid and suggestions:
            validation_status = "warning"

        logger.info(
            f"Successfully generated and stored GMN {stored_spec.id} v{stored_spec.version_number}"
        )

        return GMNGenerationResponse(
            gmn_specification=gmn_spec,
            specification_id=str(stored_spec.id),
            agent_id=str(agent_uuid),
            version_number=stored_spec.version_number,
            validation_status=validation_status,
            validation_errors=validation_errors,
            suggestions=suggestions,
            metadata={
                "node_count": stored_spec.node_count,
                "edge_count": stored_spec.edge_count,
                "complexity_score": stored_spec.complexity_score,
                "checksum": stored_spec.specification_checksum,
            },
        )

    except ValueError as e:
        logger.error(f"Invalid input for GMN generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to generate GMN: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GMN generation failed. Please try again.",
        )


@router.post("/validate", response_model=GMNValidationResponse)
async def validate_gmn(
    request: GMNValidationRequest,
    generator: GMNGenerator = Depends(get_gmn_generator),
) -> GMNValidationResponse:
    """Validate a GMN specification.

    This endpoint validates an existing GMN specification and provides
    suggestions for improvement.
    """
    try:
        logger.info("Validating GMN specification...")

        # Validate the GMN
        is_valid, errors = await generator.validate_gmn(request.gmn_specification)

        # Get improvement suggestions
        suggestions = await generator.suggest_improvements(request.gmn_specification)

        return GMNValidationResponse(is_valid=is_valid, errors=errors, suggestions=suggestions)

    except Exception as e:
        logger.error(f"Failed to validate GMN: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GMN validation failed. Please try again.",
        )


@router.post("/refine", response_model=GMNGenerationResponse)
async def refine_gmn(
    request: GMNRefinementRequest,
    generator: GMNGenerator = Depends(get_gmn_generator),
    repository: GMNVersionedRepository = Depends(get_gmn_repository),
) -> GMNGenerationResponse:
    """Refine a GMN specification based on feedback.

    This endpoint takes an existing GMN specification and feedback,
    then generates an improved version.
    """
    try:
        logger.info("Refining GMN specification based on feedback...")

        # Refine the GMN
        refined_gmn = await generator.refine_gmn(request.gmn_specification, request.feedback)

        # Validate the refined GMN
        is_valid, validation_errors = await generator.validate_gmn(refined_gmn)

        # Get improvement suggestions
        suggestions = await generator.suggest_improvements(refined_gmn)

        # For refinement, we don't automatically store - client can choose to store via /generate

        # Determine validation status
        validation_status = "valid" if is_valid else "invalid"
        if is_valid and suggestions:
            validation_status = "warning"

        return GMNGenerationResponse(
            gmn_specification=refined_gmn,
            specification_id="",  # Not stored yet
            agent_id="",  # Not stored yet
            version_number=0,  # Not stored yet
            validation_status=validation_status,
            validation_errors=validation_errors,
            suggestions=suggestions,
            metadata={"refined": True},
        )

    except ValueError as e:
        logger.error(f"Invalid input for GMN refinement: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid input: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to refine GMN: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GMN refinement failed. Please try again.",
        )


def _parse_gmn_basic(gmn_spec: str) -> Dict[str, Any]:
    """Basic GMN parsing for storage metrics.

    Args:
        gmn_spec: GMN specification string

    Returns:
        Dictionary with parsed node and edge information
    """
    lines = gmn_spec.strip().split("\n")
    nodes = []
    edges = []

    current_node = None

    for line in lines:
        line = line.strip()

        # Detect node definitions
        if line.startswith("node "):
            parts = line.split()
            if len(parts) >= 3:
                node_type = parts[1]
                node_name = parts[2].rstrip("{")
                current_node = {"name": node_name, "type": node_type, "properties": {}}
                nodes.append(current_node)

        # Detect properties
        elif current_node and ":" in line and not line.startswith("//"):
            if "from:" in line or "to:" in line:
                # This indicates an edge/relationship
                edge_info = {
                    "from_node": current_node["name"],
                    "type": line.split(":")[0].strip(),
                    "target": line.split(":")[1].strip(),
                }
                edges.append(edge_info)
            else:
                # Regular property
                prop_parts = line.split(":", 1)
                if len(prop_parts) == 2:
                    prop_name = prop_parts[0].strip()
                    prop_value = prop_parts[1].strip()
                    current_node["properties"][prop_name] = prop_value

    return {"nodes": nodes, "edges": edges, "parsed_timestamp": "utc_now"}

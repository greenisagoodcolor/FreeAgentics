"""Inference API endpoints for Active Inference agents.

This module provides REST API endpoints for inference operations,
including belief updates, action selection, and model queries.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth.security_implementation import get_current_user
from agents.inference_engine import InferenceEngine, InferenceError
from agents.pymdp_agent_factory import PyMDPAgentFactory, PyMDPAgentCreationError

import numpy as np

logger = logging.getLogger(__name__)


def convert_numpy_to_python(obj):
    """Recursively convert numpy arrays and scalars to Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_to_python(item) for item in obj)
    else:
        return obj


# Create singleton instances for inference operations
_inference_engine = InferenceEngine()
_agent_factory = PyMDPAgentFactory()

# Create router instance
router = APIRouter(
    prefix="/inference",
    tags=["inference"],
    responses={404: {"description": "Not found"}},
)


class InferenceRequest(BaseModel):
    """Request model for inference operations."""

    agent_spec: Dict[str, Any]  # GMN specification for creating agent
    observation: List[int]  # Observation as list of integers
    planning_horizon: Optional[int] = None
    timeout_ms: Optional[int] = None
    context: Optional[Dict[str, Any]] = None


class InferenceResponse(BaseModel):
    """Response model for inference operations."""

    action: Optional[Any] = None
    beliefs: Optional[Dict[str, Any]] = None
    free_energy: float = 0.0
    confidence: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class BatchInferenceRequest(BaseModel):
    """Request model for batch inference operations."""

    agent_spec: Dict[str, Any]
    observations: List[List[int]]
    timeout_ms: Optional[int] = None


class BatchInferenceResponse(BaseModel):
    """Response model for batch inference operations."""

    results: List[InferenceResponse]
    metadata: Optional[Dict[str, Any]] = None


class ModelQueryRequest(BaseModel):
    """Request model for querying agent models."""

    agent_id: str
    query_type: str
    parameters: Optional[Dict[str, Any]] = None


class ModelQueryResponse(BaseModel):
    """Response model for model queries."""

    agent_id: str
    query_type: str
    result: Any
    metadata: Optional[Dict[str, Any]] = None


@router.post("/run_inference", response_model=InferenceResponse)
async def run_inference(
    request: InferenceRequest, current_user: Dict = Depends(get_current_user)
) -> InferenceResponse:
    """Run full inference loop: belief update + action selection.

    Args:
        request: Inference request with agent spec and observation
        current_user: Authenticated user from JWT

    Returns:
        InferenceResponse with action, beliefs, and metadata
    """
    logger.info(f"Running inference with observation: {request.observation}")

    try:
        # Create PyMDP agent from specification
        agent = _agent_factory.create_agent(request.agent_spec)
        
        # Run inference
        result = _inference_engine.run_inference(
            agent=agent,
            observation=request.observation,
            planning_horizon=request.planning_horizon,
            timeout_ms=request.timeout_ms
        )
        
        if result is None:
            raise HTTPException(
                status_code=408,
                detail="Inference operation timed out or was cancelled"
            )
        
        # Convert all numpy arrays to Python types for JSON serialization
        return InferenceResponse(
            action=convert_numpy_to_python(result.action),
            beliefs=convert_numpy_to_python(result.beliefs),
            free_energy=convert_numpy_to_python(result.free_energy),
            confidence=convert_numpy_to_python(result.confidence),
            metadata=convert_numpy_to_python(result.metadata)
        )
        
    except InferenceError as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except PyMDPAgentCreationError as e:
        logger.error(f"Agent creation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error during inference: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/batch_inference", response_model=BatchInferenceResponse)
async def batch_inference(
    request: BatchInferenceRequest, current_user: Dict = Depends(get_current_user)
) -> BatchInferenceResponse:
    """Run inference on multiple observations in batch.

    Args:
        request: Batch inference request with agent spec and observations
        current_user: Authenticated user from JWT

    Returns:
        BatchInferenceResponse with results for each observation
    """
    logger.info(f"Running batch inference on {len(request.observations)} observations")

    try:
        # Create PyMDP agent from specification
        agent = _agent_factory.create_agent(request.agent_spec)
        
        # Run batch inference
        results = _inference_engine.run_batch_inference(
            agent=agent,
            observations=request.observations,
            timeout_ms=request.timeout_ms
        )
        
        # Convert to response format
        response_results = []
        for result in results:
            if result.action is not None:  # Successful inference
                response_results.append(InferenceResponse(
                    action=convert_numpy_to_python(result.action),
                    beliefs=convert_numpy_to_python(result.beliefs),
                    free_energy=convert_numpy_to_python(result.free_energy),
                    confidence=convert_numpy_to_python(result.confidence),
                    metadata=convert_numpy_to_python(result.metadata)
                ))
            else:  # Failed inference
                response_results.append(InferenceResponse(
                    action=None,
                    beliefs={},
                    free_energy=float('inf'),
                    confidence=0.0,
                    metadata={"failed": True}
                ))
        
        return BatchInferenceResponse(
            results=response_results,
            metadata={
                "total_observations": len(request.observations),
                "successful_inferences": sum(1 for r in results if r.action is not None)
            }
        )
        
    except PyMDPAgentCreationError as e:
        logger.error(f"Agent creation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query_model", response_model=ModelQueryResponse)
async def query_model(
    request: ModelQueryRequest, current_user: Dict = Depends(get_current_user)
) -> ModelQueryResponse:
    """Query agent's generative model.

    Args:
        request: Model query request
        current_user: Authenticated user from JWT

    Returns:
        ModelQueryResponse with query results
    """
    logger.info(f"Querying model for agent {request.agent_id}, type: {request.query_type}")

    # TODO: Implement actual model query logic
    # For now, return mock response based on query type
    result = {}
    if request.query_type == "parameters":
        result = {
            "A_matrix_shape": [10, 10],
            "B_matrix_shape": [10, 10, 4],
            "C_vector": [0.0] * 10,
            "D_vector": [0.25, 0.25, 0.25, 0.25],
        }
    elif request.query_type == "free_energy":
        result = {"variational_free_energy": -3.14, "expected_free_energy": -2.71}

    return ModelQueryResponse(
        agent_id=request.agent_id,
        query_type=request.query_type,
        result=result,
        metadata={"model_version": "1.0.0", "pymdp_version": "0.0.1"},
    )


@router.get("/metrics")
async def get_inference_metrics(
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get inference engine performance metrics.

    Args:
        current_user: Authenticated user from JWT

    Returns:
        Dictionary with performance metrics
    """
    engine_metrics = _inference_engine.get_metrics()
    factory_metrics = _agent_factory.get_metrics()
    
    return {
        "inference_engine": engine_metrics,
        "agent_factory": factory_metrics,
        "timestamp": "2024-01-01T00:00:00Z"  # TODO: Add real timestamp
    }


@router.get("/health")
async def inference_health() -> Dict[str, Any]:
    """Health check endpoint for inference service.

    Returns:
        Dictionary with health status
    """
    logger.debug("Inference health check")
    
    try:
        # Test PyMDP availability by checking imports
        from pymdp.agent import Agent as PyMDPAgent
        pymdp_available = True
        pymdp_status = "available"
    except ImportError:
        pymdp_available = False
        pymdp_status = "not_available"
    
    # Get basic metrics
    engine_metrics = _inference_engine.get_metrics()
    factory_metrics = _agent_factory.get_metrics()
    
    return {
        "status": "healthy",
        "service": "inference",
        "pymdp_available": pymdp_available,
        "pymdp_status": pymdp_status,
        "total_inferences": engine_metrics.get("inferences_completed", 0),
        "total_agents_created": factory_metrics.get("agents_created", 0),
        "inference_success_rate": engine_metrics.get("success_rate", 0.0)
    }


# Export router and routes for testing
__all__ = ["router"]

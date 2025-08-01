"""
Belief Monitoring API Endpoints

Provides REST API endpoints for monitoring and visualizing agent belief states.
Enables real-time belief state inspection, historical analysis, and system health monitoring.

Following Nemesis Committee consensus:
- Sarah Drasner: User-facing API with real-time updates and graceful degradation
- Jessica Kerr: Comprehensive observability with structured metrics
- Addy Osmani: Performance-optimized with caching and async processing
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.v1.services.belief_persistence import BeliefPersistenceConfig, create_belief_repository
from api.v1.services.pymdp_belief_manager import get_belief_manager

logger = logging.getLogger(__name__)

# Create router for belief monitoring endpoints
router = APIRouter(prefix="/api/v1/beliefs", tags=["belief-monitoring"])


# Pydantic models for API responses
class BeliefMetrics(BaseModel):
    """Metrics for a single belief state."""

    entropy: float = Field(description="Shannon entropy of belief distribution")
    max_confidence: float = Field(description="Maximum confidence in any single state")
    effective_states: int = Field(description="Number of states with significant probability")
    most_likely_state: int = Field(description="Index of most likely state")
    timestamp: float = Field(description="Timestamp when belief was recorded")


class BeliefStateResponse(BaseModel):
    """Complete belief state information."""

    agent_id: str
    beliefs: List[float] = Field(description="Probability distribution over states")
    metrics: BeliefMetrics
    observation_history: List[int] = Field(description="Sequence of observations")
    observation_count: int = Field(description="Total number of observations processed")


class BeliefUpdateEvent(BaseModel):
    """Information about a belief update event."""

    agent_id: str
    observation: int
    update_time_ms: float
    entropy_change: float
    kl_divergence: float
    confidence_change: float
    timestamp: str


class AgentBeliefSummary(BaseModel):
    """Summary of belief state for an agent."""

    agent_id: str
    agent_name: Optional[str] = None
    current_metrics: BeliefMetrics
    total_updates: int
    last_update: str
    confidence_trend: str = Field(description="increasing, decreasing, or stable")
    status: str = Field(description="active, inactive, or error")


class SystemBeliefHealth(BaseModel):
    """Overall system belief health metrics."""

    total_agents: int
    active_agents: int
    total_belief_updates: int
    average_entropy: float
    average_confidence: float
    last_update: str
    error_rate: float
    performance_metrics: Dict[str, Any]


# Global belief monitoring state
_belief_repository = None
_monitoring_stats = {"total_requests": 0, "total_errors": 0, "cache_hits": 0, "cache_misses": 0}


async def get_belief_repository():
    """Dependency injection for belief repository."""
    global _belief_repository
    if _belief_repository is None:
        config = BeliefPersistenceConfig()
        _belief_repository = create_belief_repository(config)
        if hasattr(_belief_repository, "initialize"):
            await _belief_repository.initialize()
    return _belief_repository


@router.get("/agent/{agent_id}", response_model=BeliefStateResponse)
async def get_agent_belief_state(
    agent_id: str, repository=Depends(get_belief_repository)
) -> BeliefStateResponse:
    """Get current belief state for a specific agent.

    Args:
        agent_id: Unique identifier for the agent

    Returns:
        Current belief state with metrics and history

    Raises:
        HTTPException: If agent not found or belief state unavailable
    """
    try:
        _monitoring_stats["total_requests"] += 1

        # Try to get belief manager first (for active agents)
        belief_manager = get_belief_manager(agent_id)

        if belief_manager:
            # Get current beliefs from active manager
            current_beliefs = belief_manager.get_current_beliefs()

            response = BeliefStateResponse(
                agent_id=agent_id,
                beliefs=current_beliefs.beliefs.tolist(),
                metrics=BeliefMetrics(
                    entropy=current_beliefs.entropy,
                    max_confidence=current_beliefs.max_confidence,
                    effective_states=current_beliefs.effective_states,
                    most_likely_state=current_beliefs.most_likely_state(),
                    timestamp=current_beliefs.timestamp,
                ),
                observation_history=current_beliefs.observation_history,
                observation_count=len(current_beliefs.observation_history),
            )

            _monitoring_stats["cache_hits"] += 1
            return response

        # Fall back to persistent storage
        current_belief = await repository.get_current_belief_state(agent_id)

        if not current_belief:
            _monitoring_stats["total_errors"] += 1
            raise HTTPException(
                status_code=404, detail=f"No belief state found for agent {agent_id}"
            )

        response = BeliefStateResponse(
            agent_id=agent_id,
            beliefs=current_belief.beliefs.tolist(),
            metrics=BeliefMetrics(
                entropy=current_belief.entropy,
                max_confidence=current_belief.max_confidence,
                effective_states=current_belief.effective_states,
                most_likely_state=current_belief.most_likely_state(),
                timestamp=current_belief.timestamp,
            ),
            observation_history=current_belief.observation_history,
            observation_count=len(current_belief.observation_history),
        )

        _monitoring_stats["cache_misses"] += 1
        return response

    except HTTPException:
        raise
    except Exception as e:
        _monitoring_stats["total_errors"] += 1
        logger.error(f"Failed to get belief state for agent {agent_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal error retrieving belief state: {str(e)}"
        )


@router.get("/agent/{agent_id}/history", response_model=List[BeliefStateResponse])
async def get_agent_belief_history(
    agent_id: str,
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of historical states"),
    repository=Depends(get_belief_repository),
) -> List[BeliefStateResponse]:
    """Get historical belief states for an agent.

    Args:
        agent_id: Unique identifier for the agent
        limit: Maximum number of historical states to return

    Returns:
        List of historical belief states ordered by timestamp (newest first)
    """
    try:
        _monitoring_stats["total_requests"] += 1

        history = await repository.get_belief_history(agent_id, limit)

        responses = []
        for belief_state in history:
            response = BeliefStateResponse(
                agent_id=agent_id,
                beliefs=belief_state.beliefs.tolist(),
                metrics=BeliefMetrics(
                    entropy=belief_state.entropy,
                    max_confidence=belief_state.max_confidence,
                    effective_states=belief_state.effective_states,
                    most_likely_state=belief_state.most_likely_state(),
                    timestamp=belief_state.timestamp,
                ),
                observation_history=belief_state.observation_history,
                observation_count=len(belief_state.observation_history),
            )
            responses.append(response)

        return responses

    except Exception as e:
        _monitoring_stats["total_errors"] += 1
        logger.error(f"Failed to get belief history for agent {agent_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal error retrieving belief history: {str(e)}"
        )


@router.get("/agent/{agent_id}/metrics", response_model=Dict[str, Any])
async def get_agent_belief_metrics(
    agent_id: str, repository=Depends(get_belief_repository)
) -> Dict[str, Any]:
    """Get comprehensive belief metrics for an agent.

    Args:
        agent_id: Unique identifier for the agent

    Returns:
        Dictionary with detailed belief metrics and performance data
    """
    try:
        _monitoring_stats["total_requests"] += 1

        # Get belief manager for active metrics
        belief_manager = get_belief_manager(agent_id)

        if belief_manager:
            # Get comprehensive metrics from active manager
            health_metrics = belief_manager.get_health_metrics()
            belief_summary = belief_manager.get_belief_summary()

            # Combine metrics
            metrics = {
                **health_metrics,
                **belief_summary,
                "source": "active_manager",
                "retrieved_at": datetime.now().isoformat(),
            }

            return metrics

        # Fall back to repository-based metrics
        current_belief = await repository.get_current_belief_state(agent_id)
        history = await repository.get_belief_history(agent_id, limit=10)

        if not current_belief:
            raise HTTPException(
                status_code=404, detail=f"No belief metrics available for agent {agent_id}"
            )

        # Calculate basic metrics from history
        entropy_values = [b.entropy for b in history]
        confidence_values = [b.max_confidence for b in history]

        metrics = {
            "agent_id": agent_id,
            "current_entropy": current_belief.entropy,
            "current_confidence": current_belief.max_confidence,
            "current_most_likely_state": current_belief.most_likely_state(),
            "total_observations": len(current_belief.observation_history),
            "historical_states": len(history),
            "average_entropy": sum(entropy_values) / len(entropy_values) if entropy_values else 0.0,
            "average_confidence": sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0.0,
            "entropy_trend": _calculate_trend(entropy_values),
            "confidence_trend": _calculate_trend(confidence_values),
            "source": "repository",
            "retrieved_at": datetime.now().isoformat(),
        }

        return metrics

    except HTTPException:
        raise
    except Exception as e:
        _monitoring_stats["total_errors"] += 1
        logger.error(f"Failed to get belief metrics for agent {agent_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal error retrieving belief metrics: {str(e)}"
        )


@router.get("/summary", response_model=List[AgentBeliefSummary])
async def get_all_agents_belief_summary(
    active_only: bool = Query(default=False, description="Return only currently active agents"),
    repository=Depends(get_belief_repository),
) -> List[AgentBeliefSummary]:
    """Get belief summary for all agents in the system.

    Args:
        active_only: If True, return only agents with active belief managers

    Returns:
        List of agent belief summaries
    """
    try:
        _monitoring_stats["total_requests"] += 1

        summaries = []

        # Get active agents from belief managers
        from api.v1.services.pymdp_belief_manager import _belief_managers

        for agent_id, manager in _belief_managers.items():
            try:
                health_metrics = manager.get_health_metrics()
                current_beliefs = manager.get_current_beliefs()

                # Calculate confidence trend (simplified)
                confidence_trend = "stable"  # Would need history for proper calculation

                summary = AgentBeliefSummary(
                    agent_id=agent_id,
                    current_metrics=BeliefMetrics(
                        entropy=current_beliefs.entropy,
                        max_confidence=current_beliefs.max_confidence,
                        effective_states=current_beliefs.effective_states,
                        most_likely_state=current_beliefs.most_likely_state(),
                        timestamp=current_beliefs.timestamp,
                    ),
                    total_updates=health_metrics.get("total_updates", 0),
                    last_update=datetime.fromtimestamp(current_beliefs.timestamp).isoformat(),
                    confidence_trend=confidence_trend,
                    status="active",
                )

                summaries.append(summary)

            except Exception as e:
                logger.warning(f"Failed to get summary for active agent {agent_id}: {e}")

        # If not active_only, also get from repository
        if not active_only:
            # This would require additional repository methods to list all agents
            # For now, just return active agents
            pass

        return summaries

    except Exception as e:
        _monitoring_stats["total_errors"] += 1
        logger.error(f"Failed to get agents belief summary: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal error retrieving belief summary: {str(e)}"
        )


@router.get("/health", response_model=SystemBeliefHealth)
async def get_system_belief_health(repository=Depends(get_belief_repository)) -> SystemBeliefHealth:
    """Get overall system belief health and performance metrics.

    Returns:
        System-wide belief health metrics
    """
    try:
        _monitoring_stats["total_requests"] += 1

        # Get active agents count
        from api.v1.services.pymdp_belief_manager import _belief_managers

        active_agents = len(_belief_managers)

        # Calculate system-wide metrics
        total_updates = 0
        entropy_values = []
        confidence_values = []
        last_update_timestamp = 0

        for manager in _belief_managers.values():
            try:
                health_metrics = manager.get_health_metrics()
                current_beliefs = manager.get_current_beliefs()

                total_updates += health_metrics.get("total_updates", 0)
                entropy_values.append(current_beliefs.entropy)
                confidence_values.append(current_beliefs.max_confidence)
                last_update_timestamp = max(last_update_timestamp, current_beliefs.timestamp)

            except Exception as e:
                logger.warning(f"Failed to get metrics for agent health calculation: {e}")

        # Calculate averages
        avg_entropy = sum(entropy_values) / len(entropy_values) if entropy_values else 0.0
        avg_confidence = (
            sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        )

        # Calculate error rate
        total_requests = max(_monitoring_stats["total_requests"], 1)
        error_rate = _monitoring_stats["total_errors"] / total_requests

        # Get repository stats if available
        repo_stats = {}
        if hasattr(repository, "get_persistence_stats"):
            try:
                repo_stats = await repository.get_persistence_stats()
            except Exception as e:
                logger.warning(f"Failed to get repository stats: {e}")

        health = SystemBeliefHealth(
            total_agents=active_agents,  # This could be enhanced with repository data
            active_agents=active_agents,
            total_belief_updates=total_updates,
            average_entropy=avg_entropy,
            average_confidence=avg_confidence,
            last_update=datetime.fromtimestamp(last_update_timestamp).isoformat()
            if last_update_timestamp
            else "never",
            error_rate=error_rate,
            performance_metrics={
                "api_requests": _monitoring_stats["total_requests"],
                "api_errors": _monitoring_stats["total_errors"],
                "cache_hit_rate": _monitoring_stats["cache_hits"] / max(total_requests, 1),
                "repository_stats": repo_stats,
                "monitoring_uptime": "active",  # Could track actual uptime
            },
        )

        return health

    except Exception as e:
        _monitoring_stats["total_errors"] += 1
        logger.error(f"Failed to get system belief health: {e}")
        raise HTTPException(
            status_code=500, detail=f"Internal error retrieving system health: {str(e)}"
        )


@router.post("/agent/{agent_id}/reset")
async def reset_agent_beliefs(
    agent_id: str, repository=Depends(get_belief_repository)
) -> JSONResponse:
    """Reset belief state for an agent to initial uniform distribution.

    Args:
        agent_id: Unique identifier for the agent

    Returns:
        Success message with reset confirmation
    """
    try:
        _monitoring_stats["total_requests"] += 1

        # Try to reset active manager first
        belief_manager = get_belief_manager(agent_id)

        if belief_manager:
            belief_manager.get_belief_manager().reset_beliefs()

            logger.info(f"Reset beliefs for active agent {agent_id}")
            return JSONResponse(
                content={
                    "message": f"Successfully reset beliefs for agent {agent_id}",
                    "agent_id": agent_id,
                    "reset_type": "active_manager",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Fall back to clearing repository history
        await repository.clear_history(agent_id)

        logger.info(f"Cleared belief history for agent {agent_id}")
        return JSONResponse(
            content={
                "message": f"Successfully cleared belief history for agent {agent_id}",
                "agent_id": agent_id,
                "reset_type": "repository_clear",
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        _monitoring_stats["total_errors"] += 1
        logger.error(f"Failed to reset beliefs for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error resetting beliefs: {str(e)}")


@router.get("/monitoring/stats")
async def get_monitoring_stats() -> Dict[str, Any]:
    """Get API monitoring statistics.

    Returns:
        Dictionary with API usage and performance statistics
    """
    return {
        "total_requests": _monitoring_stats["total_requests"],
        "total_errors": _monitoring_stats["total_errors"],
        "cache_hits": _monitoring_stats["cache_hits"],
        "cache_misses": _monitoring_stats["cache_misses"],
        "error_rate": _monitoring_stats["total_errors"]
        / max(_monitoring_stats["total_requests"], 1),
        "cache_hit_rate": _monitoring_stats["cache_hits"]
        / max(_monitoring_stats["total_requests"], 1),
        "uptime": "active",  # Could track actual uptime
        "timestamp": datetime.now().isoformat(),
    }


def _calculate_trend(values: List[float]) -> str:
    """Calculate trend direction from a list of values.

    Args:
        values: List of numeric values ordered by time

    Returns:
        "increasing", "decreasing", or "stable"
    """
    if len(values) < 2:
        return "stable"

    # Simple linear trend calculation
    first_half = values[: len(values) // 2]
    second_half = values[len(values) // 2 :]

    first_avg = sum(first_half) / len(first_half)
    second_avg = sum(second_half) / len(second_half)

    diff = second_avg - first_avg
    threshold = 0.05  # 5% change threshold

    if diff > threshold:
        return "increasing"
    elif diff < -threshold:
        return "decreasing"
    else:
        return "stable"

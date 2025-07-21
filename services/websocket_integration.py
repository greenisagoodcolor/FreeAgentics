"""WebSocket integration helper for real-time pipeline updates.

This module provides utilities for sending real-time updates during the
prompt → agent → knowledge graph pipeline processing.
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Enumeration of pipeline processing stages."""

    INITIALIZATION = "initialization"
    GMN_GENERATION = "gmn_generation"
    GMN_VALIDATION = "gmn_validation"
    AGENT_CREATION = "agent_creation"
    DATABASE_STORAGE = "database_storage"
    KNOWLEDGE_GRAPH_UPDATE = "knowledge_graph_update"
    SUGGESTION_GENERATION = "suggestion_generation"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineEventType(Enum):
    """Types of pipeline events for WebSocket broadcasting."""

    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_PROGRESS = "pipeline_progress"
    GMN_GENERATED = "gmn_generated"
    VALIDATION_SUCCESS = "validation_success"
    VALIDATION_FAILED = "validation_failed"
    AGENT_CREATED = "agent_created"
    KNOWLEDGE_GRAPH_UPDATED = "knowledge_graph_updated"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_FAILED = "pipeline_failed"
    STAGE_ERROR = "stage_error"
    STAGE_WARNING = "stage_warning"


class WebSocketPipelineMonitor:
    """Monitor and broadcast pipeline processing events via WebSocket."""

    def __init__(self):
        """Initialize the pipeline monitor."""
        self.active_pipelines: Dict[str, Dict[str, Any]] = {}
        self.pipeline_subscribers: Dict[str, Set[str]] = {}

    def start_pipeline(
        self, prompt_id: str, user_id: str, prompt_text: str
    ) -> Dict[str, Any]:
        """Record the start of a new pipeline processing.

        Args:
            prompt_id: Unique identifier for the prompt
            user_id: User who initiated the prompt
            prompt_text: The original prompt text

        Returns:
            Pipeline tracking information
        """
        pipeline_info = {
            "prompt_id": prompt_id,
            "user_id": user_id,
            "prompt_text": prompt_text[:100] + "..."
            if len(prompt_text) > 100
            else prompt_text,
            "start_time": datetime.utcnow(),
            "current_stage": PipelineStage.INITIALIZATION.value,
            "stages_completed": [],
            "errors": [],
            "warnings": [],
            "metrics": {},
        }

        self.active_pipelines[prompt_id] = pipeline_info
        logger.info(f"Started tracking pipeline for prompt {prompt_id}")

        return pipeline_info

    def update_stage(
        self,
        prompt_id: str,
        stage: PipelineStage,
        message: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update the current stage of pipeline processing.

        Args:
            prompt_id: Prompt identifier
            stage: Current pipeline stage
            message: Optional status message
            data: Optional additional data

        Returns:
            Updated pipeline information
        """
        if prompt_id not in self.active_pipelines:
            logger.warning(f"Prompt {prompt_id} not found in active pipelines")
            return {}

        pipeline = self.active_pipelines[prompt_id]
        previous_stage = pipeline["current_stage"]

        # Update stage
        pipeline["current_stage"] = stage.value
        pipeline["last_update"] = datetime.utcnow()

        # Record completed stage
        if (
            previous_stage != stage.value
            and previous_stage != PipelineStage.INITIALIZATION.value
        ):
            pipeline["stages_completed"].append(
                {
                    "stage": previous_stage,
                    "completed_at": datetime.utcnow().isoformat(),
                    "duration_ms": self._calculate_stage_duration(
                        pipeline, previous_stage
                    ),
                }
            )

        # Add stage-specific data
        if data:
            pipeline["metrics"][stage.value] = data

        logger.info(f"Pipeline {prompt_id} moved to stage: {stage.value}")

        return {
            "prompt_id": prompt_id,
            "current_stage": stage.value,
            "message": message,
            "stages_completed": len(pipeline["stages_completed"]),
            "total_stages": 6,
            "data": data,
        }

    def record_error(
        self,
        prompt_id: str,
        error: str,
        stage: PipelineStage,
        error_type: str = "error",
    ) -> Dict[str, Any]:
        """Record an error during pipeline processing.

        Args:
            prompt_id: Prompt identifier
            error: Error message
            stage: Stage where error occurred
            error_type: Type of error

        Returns:
            Error information
        """
        if prompt_id not in self.active_pipelines:
            return {}

        pipeline = self.active_pipelines[prompt_id]
        error_info = {
            "error": error,
            "stage": stage.value,
            "type": error_type,
            "timestamp": datetime.utcnow().isoformat(),
        }

        pipeline["errors"].append(error_info)
        pipeline["current_stage"] = PipelineStage.FAILED.value

        logger.error(f"Pipeline {prompt_id} error at {stage.value}: {error}")

        return error_info

    def complete_pipeline(
        self,
        prompt_id: str,
        agent_id: str,
        processing_time_ms: float,
        result_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Mark pipeline as completed successfully.

        Args:
            prompt_id: Prompt identifier
            agent_id: Created agent ID
            processing_time_ms: Total processing time
            result_data: Final result data

        Returns:
            Completion information
        """
        if prompt_id not in self.active_pipelines:
            return {}

        pipeline = self.active_pipelines[prompt_id]
        pipeline["current_stage"] = PipelineStage.COMPLETED.value
        pipeline["end_time"] = datetime.utcnow()
        pipeline["agent_id"] = agent_id
        pipeline["processing_time_ms"] = processing_time_ms
        pipeline["result"] = result_data

        # Calculate stage metrics
        stage_timings = self._calculate_stage_timings(pipeline)

        completion_info = {
            "prompt_id": prompt_id,
            "agent_id": agent_id,
            "processing_time_ms": processing_time_ms,
            "stages_completed": len(pipeline["stages_completed"]) + 1,
            "stage_timings": stage_timings,
            "result_summary": {
                "suggestions_count": len(result_data.get("next_suggestions", [])),
                "kg_updates_count": len(result_data.get("knowledge_graph_updates", [])),
            },
        }

        logger.info(f"Pipeline {prompt_id} completed in {processing_time_ms}ms")

        # Clean up after a delay
        asyncio.create_task(self._cleanup_pipeline(prompt_id, delay=300))  # 5 minutes

        return completion_info

    def get_pipeline_status(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a pipeline.

        Args:
            prompt_id: Prompt identifier

        Returns:
            Pipeline status or None if not found
        """
        return self.active_pipelines.get(prompt_id)

    def get_active_pipelines(
        self, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all active pipelines, optionally filtered by user.

        Args:
            user_id: Optional user filter

        Returns:
            List of active pipeline information
        """
        pipelines = []
        for prompt_id, pipeline in self.active_pipelines.items():
            if user_id is None or pipeline.get("user_id") == user_id:
                pipelines.append(
                    {
                        "prompt_id": prompt_id,
                        "user_id": pipeline.get("user_id"),
                        "current_stage": pipeline.get("current_stage"),
                        "start_time": pipeline.get("start_time").isoformat()
                        if pipeline.get("start_time")
                        else None,
                        "errors_count": len(pipeline.get("errors", [])),
                        "stages_completed": len(pipeline.get("stages_completed", [])),
                    }
                )
        return pipelines

    def subscribe_to_pipeline(self, prompt_id: str, client_id: str):
        """Subscribe a WebSocket client to pipeline updates.

        Args:
            prompt_id: Prompt identifier
            client_id: WebSocket client ID
        """
        if prompt_id not in self.pipeline_subscribers:
            self.pipeline_subscribers[prompt_id] = set()
        self.pipeline_subscribers[prompt_id].add(client_id)
        logger.info(f"Client {client_id} subscribed to pipeline {prompt_id}")

    def unsubscribe_from_pipeline(self, prompt_id: str, client_id: str):
        """Unsubscribe a WebSocket client from pipeline updates.

        Args:
            prompt_id: Prompt identifier
            client_id: WebSocket client ID
        """
        if prompt_id in self.pipeline_subscribers:
            self.pipeline_subscribers[prompt_id].discard(client_id)
            if not self.pipeline_subscribers[prompt_id]:
                del self.pipeline_subscribers[prompt_id]

    def get_pipeline_subscribers(self, prompt_id: str) -> Set[str]:
        """Get all subscribers for a pipeline.

        Args:
            prompt_id: Prompt identifier

        Returns:
            Set of client IDs subscribed to this pipeline
        """
        return self.pipeline_subscribers.get(prompt_id, set())

    def _calculate_stage_duration(self, pipeline: Dict[str, Any], stage: str) -> float:
        """Calculate duration of a stage in milliseconds."""
        # This is a simplified calculation
        # In production, you'd track actual stage start/end times
        return 0.0

    def _calculate_stage_timings(self, pipeline: Dict[str, Any]) -> Dict[str, float]:
        """Calculate timing breakdown by stage."""
        timings = {}
        total_time = pipeline.get("processing_time_ms", 0)
        num_stages = len(pipeline.get("stages_completed", [])) + 1

        # Simple even distribution for now
        # In production, track actual timings
        if num_stages > 0:
            avg_time = total_time / num_stages
            for stage in PipelineStage:
                if stage not in [
                    PipelineStage.INITIALIZATION,
                    PipelineStage.COMPLETED,
                    PipelineStage.FAILED,
                ]:
                    timings[stage.value] = avg_time

        return timings

    async def _cleanup_pipeline(self, prompt_id: str, delay: int = 300):
        """Clean up pipeline data after a delay.

        Args:
            prompt_id: Prompt identifier
            delay: Delay in seconds before cleanup
        """
        await asyncio.sleep(delay)

        if prompt_id in self.active_pipelines:
            del self.active_pipelines[prompt_id]
            logger.info(f"Cleaned up pipeline data for {prompt_id}")

        if prompt_id in self.pipeline_subscribers:
            del self.pipeline_subscribers[prompt_id]


# Global pipeline monitor instance
pipeline_monitor = WebSocketPipelineMonitor()


# Helper functions for easy integration
async def broadcast_pipeline_event(
    event_type: PipelineEventType,
    prompt_id: str,
    data: Dict[str, Any],
    websocket_manager=None,
):
    """Broadcast a pipeline event to WebSocket subscribers.

    Args:
        event_type: Type of pipeline event
        prompt_id: Prompt identifier
        data: Event data
        websocket_manager: Optional WebSocket manager instance
    """
    event_data = {
        "type": f"pipeline:{event_type.value}",
        "prompt_id": prompt_id,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data,
    }

    # Get subscribers for this pipeline
    subscribers = pipeline_monitor.get_pipeline_subscribers(prompt_id)

    if websocket_manager and subscribers:
        # Send to specific subscribers
        for client_id in subscribers:
            await websocket_manager.send_personal_message(event_data, client_id)
    elif websocket_manager:
        # Broadcast to all if no specific subscribers
        await websocket_manager.broadcast(
            event_data, event_type=f"pipeline:{prompt_id}"
        )

    logger.debug(f"Broadcasted {event_type.value} for pipeline {prompt_id}")


def create_pipeline_progress_message(
    stage: PipelineStage,
    stage_number: int,
    total_stages: int = 6,
    message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a standardized pipeline progress message.

    Args:
        stage: Current pipeline stage
        stage_number: Current stage number (1-based)
        total_stages: Total number of stages
        message: Optional progress message
        details: Optional additional details

    Returns:
        Formatted progress message
    """
    progress_message = {
        "stage": stage.value,
        "stage_number": stage_number,
        "total_stages": total_stages,
        "progress_percentage": (stage_number / total_stages) * 100,
        "message": message or f"Processing stage: {stage.value}",
    }

    if details:
        progress_message["details"] = details

    return progress_message

"""Knowledge Graph Update Pipeline Orchestrator (Task 59.3).

This module implements the main orchestrator for updating knowledge graphs from
PyMDP agent inference results. It integrates with existing infrastructure while
providing clean interfaces and comprehensive monitoring.

Follows Nemesis Committee architectural guidance for separation of concerns,
observability, and production readiness.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from agents.inference_engine import InferenceResult
from knowledge_graph.extraction import ExtractionResult
from knowledge_graph.graph_engine import KnowledgeGraph
from knowledge_graph.pymdp_extractor import (
    PyMDPKnowledgeExtractor,
)
from knowledge_graph.realtime_updater import (
    RealtimeGraphUpdater,
)
from observability.prometheus_metrics import (
    PrometheusMetricsCollector,
    agent_inference_duration_seconds,
    business_inference_operations_total,
)

logger = logging.getLogger(__name__)


@dataclass
class UpdateMetrics:
    """Metrics for knowledge graph update operations."""

    total_updates: int = 0
    successful_updates: int = 0
    failed_updates: int = 0
    avg_processing_time_ms: float = 0.0
    entities_created: int = 0
    relations_created: int = 0
    conflicts_resolved: int = 0

    def update_success(self, processing_time_ms: float, entities: int, relations: int):
        """Record a successful update."""
        self.total_updates += 1
        self.successful_updates += 1
        self.entities_created += entities
        self.relations_created += relations

        # Update rolling average
        if self.successful_updates == 1:
            self.avg_processing_time_ms = processing_time_ms
        else:
            alpha = 0.1  # Exponential moving average factor
            self.avg_processing_time_ms = (
                alpha * processing_time_ms + (1 - alpha) * self.avg_processing_time_ms
            )

    def update_failure(self):
        """Record a failed update."""
        self.total_updates += 1
        self.failed_updates += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_updates": self.total_updates,
            "successful_updates": self.successful_updates,
            "failed_updates": self.failed_updates,
            "success_rate": self.successful_updates / max(1, self.total_updates),
            "avg_processing_time_ms": self.avg_processing_time_ms,
            "entities_created": self.entities_created,
            "relations_created": self.relations_created,
            "conflicts_resolved": self.conflicts_resolved,
        }


@dataclass
class UpdateRequest:
    """Request for knowledge graph update from inference result."""

    inference_result: InferenceResult
    agent_id: str
    conversation_id: str
    message_id: Optional[str] = None
    trace_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize computed fields."""
        if self.message_id is None:
            self.message_id = str(uuid4())
        if self.trace_id is None:
            self.trace_id = str(uuid4())


class KnowledgeGraphUpdater:
    """Main orchestrator for PyMDP knowledge graph updates.

    This class coordinates between PyMDP inference results and knowledge graph
    updates, providing clean interfaces while maintaining separation of concerns.
    """

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        pymdp_extractor: Optional[PyMDPKnowledgeExtractor] = None,
        realtime_updater: Optional[RealtimeGraphUpdater] = None,
        batch_size: int = 10,
        batch_timeout_seconds: float = 1.0,
    ):
        """Initialize knowledge graph updater.

        Args:
            knowledge_graph: Target knowledge graph (creates new if None)
            pymdp_extractor: PyMDP knowledge extractor (creates default if None)
            realtime_updater: Real-time update system (creates default if None)
            batch_size: Maximum batch size for updates
            batch_timeout_seconds: Maximum time to wait for batch completion
        """
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.pymdp_extractor = pymdp_extractor or PyMDPKnowledgeExtractor()
        self.realtime_updater = realtime_updater or RealtimeGraphUpdater(self.knowledge_graph)

        # Batch processing configuration
        self.batch_size = batch_size
        self.batch_timeout_seconds = batch_timeout_seconds
        self.pending_updates: List[UpdateRequest] = []
        self.batch_lock = asyncio.Lock()

        # Metrics and monitoring
        self.metrics = UpdateMetrics()
        self.prometheus_metrics = PrometheusMetricsCollector()
        self.is_running = False

        # Background tasks
        self._batch_processor_task: Optional[asyncio.Task] = None

        logger.info(
            f"KnowledgeGraphUpdater initialized with batch_size={batch_size}, "
            f"timeout={batch_timeout_seconds}s"
        )

    async def start(self) -> None:
        """Start the knowledge graph updater."""
        if self.is_running:
            return

        self.is_running = True

        # Start real-time updater
        await self.realtime_updater.start()

        # Start batch processor
        self._batch_processor_task = asyncio.create_task(self._batch_processor())

        logger.info("KnowledgeGraphUpdater started")

    async def stop(self) -> None:
        """Stop the knowledge graph updater."""
        if not self.is_running:
            return

        self.is_running = False

        # Process remaining updates
        if self.pending_updates:
            await self._process_batch(self.pending_updates.copy())
            self.pending_updates.clear()

        # Stop batch processor
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                pass

        # Stop real-time updater
        await self.realtime_updater.stop()

        logger.info("KnowledgeGraphUpdater stopped")

    async def update_from_inference(
        self,
        inference_result: InferenceResult,
        agent_id: str,
        conversation_id: str,
        message_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        force_immediate: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Update knowledge graph from PyMDP inference result.

        Args:
            inference_result: Result from PyMDP inference
            agent_id: ID of the agent
            conversation_id: Conversation context ID
            message_id: Message ID (generated if None)
            trace_id: Distributed tracing ID (generated if None)
            force_immediate: Process immediately instead of batching

        Returns:
            Update result dictionary if immediate, None if batched
        """
        if not self.is_running:
            raise RuntimeError("KnowledgeGraphUpdater is not running")

        # Create update request
        request = UpdateRequest(
            inference_result=inference_result,
            agent_id=agent_id,
            conversation_id=conversation_id,
            message_id=message_id,
            trace_id=trace_id,
        )

        if force_immediate:
            # Process immediately
            return await self._process_single_update(request)
        else:
            # Add to batch
            async with self.batch_lock:
                self.pending_updates.append(request)

                # Process if batch is full
                if len(self.pending_updates) >= self.batch_size:
                    batch = self.pending_updates.copy()
                    self.pending_updates.clear()
                    # Process batch in background
                    asyncio.create_task(self._process_batch(batch))

            return None

    async def _process_single_update(self, request: UpdateRequest) -> Dict[str, Any]:
        """Process a single update request."""
        start_time = time.time()

        try:
            # Extract knowledge from inference result
            extraction_result = self.pymdp_extractor.extract_knowledge(
                inference_result=request.inference_result,
                agent_id=request.agent_id,
                conversation_id=request.conversation_id,
                message_id=request.message_id,
            )

            if extraction_result.get("metadata", {}).get("error"):
                # Extraction failed
                self.metrics.update_failure()
                logger.error(
                    f"Knowledge extraction failed: {extraction_result['metadata']['error']}"
                )
                return extraction_result

            # Create ExtractionResult for real-time updater
            entities = extraction_result["entities"]
            relations = extraction_result["relations"]

            extraction_result_obj = ExtractionResult(
                entities=entities,
                relations=relations,
                extraction_metadata=extraction_result["metadata"],
                processing_time=extraction_result["metadata"]["processing_time_seconds"],
            )

            # Process through real-time updater
            update_events = await self.realtime_updater.process_extraction_result(
                extraction_result=extraction_result_obj,
                conversation_id=request.conversation_id,
                message_id=request.message_id,
                trace_id=request.trace_id,
            )

            # Update metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self.metrics.update_success(
                processing_time_ms=processing_time_ms,
                entities=len(entities),
                relations=len(relations),
            )

            # Record Prometheus metrics
            agent_inference_duration_seconds.labels(
                agent_id=request.agent_id, operation_type="kg_update"
            ).observe(processing_time_ms / 1000.0)

            business_inference_operations_total.labels(
                operation_type="kg_update", success="true"
            ).inc()

            result = {
                **extraction_result,
                "update_events": [event.to_dict() for event in update_events],
                "processing_summary": {
                    "total_processing_time_ms": processing_time_ms,
                    "extraction_time_ms": extraction_result["metadata"]["processing_time_seconds"]
                    * 1000,
                    "update_events_count": len(update_events),
                },
            }

            logger.info(
                f"Successfully processed PyMDP update for agent {request.agent_id} "
                f"in {processing_time_ms:.2f}ms"
            )

            return result

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            self.metrics.update_failure()

            business_inference_operations_total.labels(
                operation_type="kg_update", success="false"
            ).inc()

            logger.error(
                f"Failed to process PyMDP update for agent {request.agent_id}: {e}",
                exc_info=True,
                extra={
                    "agent_id": request.agent_id,
                    "conversation_id": request.conversation_id,
                    "trace_id": request.trace_id,
                },
            )

            return {
                "entities": [],
                "relations": [],
                "metadata": {
                    "error": str(e),
                    "processing_time_ms": processing_time_ms,
                    "agent_id": request.agent_id,
                },
            }

    async def _process_batch(self, requests: List[UpdateRequest]) -> None:
        """Process a batch of update requests."""
        if not requests:
            return

        start_time = time.time()
        logger.debug(f"Processing batch of {len(requests)} updates")

        # Process all requests concurrently
        tasks = [self._process_single_update(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log batch results
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        batch_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Processed batch in {batch_time_ms:.2f}ms: "
            f"{successful} successful, {failed} failed"
        )

    async def _batch_processor(self) -> None:
        """Background task to process batches on timeout."""
        while self.is_running:
            try:
                await asyncio.sleep(self.batch_timeout_seconds)

                # Check for pending updates
                async with self.batch_lock:
                    if self.pending_updates:
                        batch = self.pending_updates.copy()
                        self.pending_updates.clear()
                        # Process batch
                        await self._process_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                await asyncio.sleep(1.0)  # Back off on error

    def get_metrics(self) -> Dict[str, Any]:
        """Get updater metrics."""
        # Create sync version of health status
        realtime_health = {
            "is_running": self.realtime_updater.is_running,
            "active_updates": len(self.realtime_updater.active_updates),
            "available_slots": self.realtime_updater.update_semaphore._value,
            "event_streamer_active": getattr(
                self.realtime_updater.event_streamer, "is_streaming", False
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return {
            "updater_metrics": self.metrics.to_dict(),
            "realtime_updater_health": realtime_health,
            "knowledge_graph_stats": {
                "node_count": len(self.knowledge_graph.nodes),
                "edge_count": len(self.knowledge_graph.edges),
                "graph_version": self.knowledge_graph.version,
            },
            "batch_status": {
                "pending_updates": len(self.pending_updates),
                "batch_size": self.batch_size,
                "batch_timeout_seconds": self.batch_timeout_seconds,
                "is_running": self.is_running,
            },
        }

    async def get_agent_knowledge(
        self, agent_id: str, entity_types: Optional[List[str]] = None, limit: int = 100
    ) -> Dict[str, Any]:
        """Get knowledge graph data for a specific agent.

        Args:
            agent_id: ID of the agent
            entity_types: Filter by entity types (all if None)
            limit: Maximum number of entities to return

        Returns:
            Dictionary with agent's knowledge graph data
        """
        try:
            # Find entities related to this agent
            agent_entities = []
            for entity in self.knowledge_graph.nodes.values():
                entity_data = entity.properties
                if entity_data.get("agent_id") == agent_id:
                    if entity_types is None or entity.type.value in entity_types:
                        agent_entities.append(
                            {
                                "id": entity.id,
                                "type": entity.type.value,
                                "label": entity.label,
                                "properties": entity_data,
                                "created_at": entity.created_at.isoformat(),
                                "updated_at": entity.updated_at.isoformat(),
                            }
                        )

            # Sort by creation time (newest first) and limit
            agent_entities.sort(key=lambda e: e["created_at"], reverse=True)
            agent_entities = agent_entities[:limit]

            # Find related edges
            entity_ids = {e["id"] for e in agent_entities}
            agent_relations = []

            for edge in self.knowledge_graph.edges.values():
                if edge.source_id in entity_ids or edge.target_id in entity_ids:
                    agent_relations.append(
                        {
                            "id": edge.id,
                            "source_id": edge.source_id,
                            "target_id": edge.target_id,
                            "type": edge.type.value,
                            "properties": edge.properties,
                            "created_at": edge.created_at.isoformat(),
                        }
                    )

            return {
                "agent_id": agent_id,
                "entities": agent_entities,
                "relations": agent_relations,
                "summary": {
                    "entity_count": len(agent_entities),
                    "relation_count": len(agent_relations),
                    "entity_types": list(set(e["type"] for e in agent_entities)),
                },
            }

        except Exception as e:
            logger.error(f"Failed to get agent knowledge for {agent_id}: {e}")
            return {
                "agent_id": agent_id,
                "error": str(e),
                "entities": [],
                "relations": [],
            }


# Convenience function for simple usage
async def update_knowledge_from_inference(
    inference_result: InferenceResult,
    agent_id: str,
    conversation_id: str,
    updater: Optional[KnowledgeGraphUpdater] = None,
) -> Dict[str, Any]:
    """Convenience function to update knowledge graph from inference result.

    Args:
        inference_result: Result from PyMDP inference
        agent_id: ID of the agent
        conversation_id: Conversation context ID
        updater: Knowledge graph updater (creates default if None)

    Returns:
        Update result dictionary
    """
    if updater is None:
        updater = KnowledgeGraphUpdater()
        await updater.start()

        try:
            result = await updater.update_from_inference(
                inference_result=inference_result,
                agent_id=agent_id,
                conversation_id=conversation_id,
                force_immediate=True,
            )
            return result
        finally:
            await updater.stop()
    else:
        return await updater.update_from_inference(
            inference_result=inference_result,
            agent_id=agent_id,
            conversation_id=conversation_id,
            force_immediate=True,
        )

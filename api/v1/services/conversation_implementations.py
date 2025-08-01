"""
Conversation Flow Control System Implementations

Implements the core interfaces required by the ConversationOrchestrator for managing
conversation turns, events, completion detection, and turn limit policies.

Following Nemesis Committee architecture with dependency injection, comprehensive
error handling, and production-ready observability.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from api.v1.models.agent_conversation import (
    AgentRole,
    CompletionReason,
    ConversationAggregate,
    ConversationContext,
    ConversationEvent,
    ConversationTurnDomain,
    IAgentResponseGenerator,
    ICompletionDetector,
    IConversationEventPublisher,
    ITurnController,
    ITurnLimitPolicy,
    TurnStatus,
)

logger = logging.getLogger(__name__)


class ConversationTurnController(ITurnController):
    """
    Production-ready turn controller that orchestrates individual conversation turns.

    Handles turn execution flow: validation, response generation, completion,
    and error handling with comprehensive observability.
    """

    def __init__(
        self,
        response_generator: IAgentResponseGenerator,
        event_publisher: Optional[IConversationEventPublisher] = None,
    ):
        """Initialize turn controller with dependencies."""
        self.response_generator = response_generator
        self.event_publisher = event_publisher

        # Turn execution metrics
        self.metrics = {
            "turns_executed": 0,
            "turns_successful": 0,
            "turns_failed": 0,
            "turns_timeout": 0,
            "avg_execution_time": 0.0,
            "total_response_length": 0,
        }

        logger.info("ConversationTurnController initialized")

    async def execute_turn(
        self, turn: ConversationTurnDomain, agent: AgentRole, context: ConversationContext
    ) -> ConversationTurnDomain:
        """
        Execute a single conversation turn with comprehensive error handling.

        Args:
            turn: Turn domain object to execute
            agent: Agent taking this turn
            context: Current conversation context

        Returns:
            Completed turn with response or error state
        """
        start_time = time.time()
        self.metrics["turns_executed"] += 1

        # Validate inputs
        self._validate_turn_inputs(turn, agent, context)

        # Mark turn as started
        turn.mark_started()

        # Publish turn start event
        await self._publish_turn_event(
            "turn_execution_started",
            {
                "turn_id": turn.turn_id,
                "turn_number": turn.turn_number,
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "conversation_id": context.conversation_id,
            },
        )

        try:
            # Generate agent response
            logger.debug(f"Generating response for turn {turn.turn_number} by {agent.name}")

            response = await self.response_generator.generate_response(
                agent=agent,
                context=context,
                timeout_seconds=30,  # Configurable timeout
            )

            # Validate response
            if not response or not response.strip():
                raise ValueError("Agent generated empty response")

            # Mark turn as completed
            turn.mark_completed(response.strip())

            # Update metrics
            execution_time = time.time() - start_time
            self._update_success_metrics(execution_time, len(response))

            # Publish turn completion event
            await self._publish_turn_event(
                "turn_execution_completed",
                {
                    "turn_id": turn.turn_id,
                    "turn_number": turn.turn_number,
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "conversation_id": context.conversation_id,
                    "response_length": len(response),
                    "execution_time_seconds": execution_time,
                    "status": turn.status.value,
                },
            )

            logger.info(
                f"Turn {turn.turn_number} completed successfully by {agent.name} "
                f"in {execution_time:.2f}s ({len(response)} chars)"
            )

            return turn

        except asyncio.TimeoutError:
            # Handle timeout
            turn.mark_timeout()
            self.metrics["turns_timeout"] += 1

            await self._publish_turn_event(
                "turn_execution_timeout",
                {
                    "turn_id": turn.turn_id,
                    "turn_number": turn.turn_number,
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "conversation_id": context.conversation_id,
                    "timeout_seconds": 30,
                },
            )

            logger.warning(f"Turn {turn.turn_number} by {agent.name} timed out")
            return turn

        except Exception as e:
            # Handle any other errors
            error_message = f"Turn execution failed: {str(e)}"
            turn.mark_failed(error_message)
            self.metrics["turns_failed"] += 1

            await self._publish_turn_event(
                "turn_execution_failed",
                {
                    "turn_id": turn.turn_id,
                    "turn_number": turn.turn_number,
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "conversation_id": context.conversation_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

            logger.error(f"Turn {turn.turn_number} by {agent.name} failed: {e}", exc_info=True)
            return turn

    def _validate_turn_inputs(
        self, turn: ConversationTurnDomain, agent: AgentRole, context: ConversationContext
    ):
        """Validate turn execution inputs."""
        if turn.status != TurnStatus.PENDING:
            raise ValueError(f"Turn must be PENDING status, got {turn.status}")

        if not agent.name or not agent.name.strip():
            raise ValueError("Agent name cannot be empty")

        if not context.conversation_id:
            raise ValueError("Context must have valid conversation ID")

        if turn.turn_number <= 0:
            raise ValueError("Turn number must be positive")

    def _update_success_metrics(self, execution_time: float, response_length: int):
        """Update success metrics with exponential moving average."""
        self.metrics["turns_successful"] += 1
        self.metrics["total_response_length"] += response_length

        # Update average execution time
        current_avg = self.metrics["avg_execution_time"]
        alpha = 0.1  # Smoothing factor
        self.metrics["avg_execution_time"] = (alpha * execution_time) + ((1 - alpha) * current_avg)

    async def _publish_turn_event(self, event_type: str, data: Dict[str, Any]):
        """Publish turn execution event if publisher available."""
        if self.event_publisher:
            try:
                event = ConversationEvent(
                    conversation_id=data.get("conversation_id", "unknown"),
                    event_type=event_type,
                    data=data,
                )
                await self.event_publisher.publish_event(event)
            except Exception as e:
                logger.warning(f"Failed to publish turn event {event_type}: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current turn controller metrics."""
        metrics = self.metrics.copy()

        # Add computed metrics
        total_turns = metrics["turns_executed"]
        if total_turns > 0:
            metrics["success_rate"] = metrics["turns_successful"] / total_turns
            metrics["failure_rate"] = metrics["turns_failed"] / total_turns
            metrics["timeout_rate"] = metrics["turns_timeout"] / total_turns
            metrics["avg_response_length"] = (
                metrics["total_response_length"] / metrics["turns_successful"]
                if metrics["turns_successful"] > 0
                else 0
            )
        else:
            metrics["success_rate"] = 0.0
            metrics["failure_rate"] = 0.0
            metrics["timeout_rate"] = 0.0
            metrics["avg_response_length"] = 0.0

        return metrics


class ConversationEventPublisher(IConversationEventPublisher):
    """
    Production-ready event publisher with multiple channels and async delivery.

    Publishes conversation events to various channels (logs, metrics, webhooks)
    with non-blocking delivery and error handling.
    """

    def __init__(self, enable_detailed_logging: bool = True):
        """Initialize event publisher."""
        self.enable_detailed_logging = enable_detailed_logging

        # Event publishing metrics
        self.metrics = {
            "events_published": 0,
            "events_successful": 0,
            "events_failed": 0,
            "events_by_type": {},
        }

        logger.info("ConversationEventPublisher initialized")

    async def publish_event(self, event: ConversationEvent) -> None:
        """
        Publish conversation event to all configured channels.

        Args:
            event: Event to publish
        """
        self.metrics["events_published"] += 1

        # Track events by type
        event_type = event.event_type
        self.metrics["events_by_type"][event_type] = (
            self.metrics["events_by_type"].get(event_type, 0) + 1
        )

        try:
            # Publish to structured logs (primary channel)
            await self._publish_to_logs(event)

            # Could add additional channels here:
            # await self._publish_to_metrics(event)
            # await self._publish_to_webhook(event)
            # await self._publish_to_message_queue(event)

            self.metrics["events_successful"] += 1

        except Exception as e:
            self.metrics["events_failed"] += 1
            logger.error(f"Failed to publish event {event.event_type}: {e}")
            # Don't re-raise - event publishing should not block conversation flow

    async def _publish_to_logs(self, event: ConversationEvent):
        """Publish event to structured logs."""
        if self.enable_detailed_logging:
            logger.info(
                f"Conversation Event: {event.event_type}",
                extra={
                    "event_id": event.event_id,
                    "conversation_id": event.conversation_id,
                    "event_type": event.event_type,
                    "timestamp": event.timestamp.isoformat(),
                    "event_data": event.data,
                },
            )
        else:
            logger.debug(f"Event: {event.event_type} for conversation {event.conversation_id}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get event publisher metrics."""
        metrics = self.metrics.copy()

        # Add computed metrics
        total_events = metrics["events_published"]
        if total_events > 0:
            metrics["success_rate"] = metrics["events_successful"] / total_events
            metrics["failure_rate"] = metrics["events_failed"] / total_events
        else:
            metrics["success_rate"] = 0.0
            metrics["failure_rate"] = 0.0

        return metrics


class TurnLimitPolicy(ITurnLimitPolicy):
    """
    Configurable turn limit policy with multiple limit strategies.

    Supports fixed turn limits, time-based limits, and quality-based limits
    with comprehensive decision logging.
    """

    def __init__(
        self,
        max_turns: int = 10,
        max_duration_minutes: int = 30,
        min_quality_threshold: float = 0.3,
        enable_quality_limits: bool = False,
    ):
        """Initialize turn limit policy with configuration."""
        self.max_turns = max_turns
        self.max_duration_minutes = max_duration_minutes
        self.min_quality_threshold = min_quality_threshold
        self.enable_quality_limits = enable_quality_limits

        # Policy decision tracking
        self.decisions = {
            "continue_decisions": 0,
            "turn_limit_stops": 0,
            "time_limit_stops": 0,
            "quality_limit_stops": 0,
        }

        logger.info(
            f"TurnLimitPolicy initialized - max_turns: {max_turns}, "
            f"max_duration: {max_duration_minutes}m, quality_limits: {enable_quality_limits}"
        )

    def should_continue(self, conversation: ConversationAggregate) -> bool:
        """
        Determine if conversation should continue based on configured policies.

        Args:
            conversation: Current conversation state

        Returns:
            True if conversation should continue, False otherwise
        """
        # Check turn count limit
        if len(conversation.turns) >= self.max_turns:
            self.decisions["turn_limit_stops"] += 1
            logger.info(
                f"Conversation {conversation.conversation_id} stopped: "
                f"turn limit reached ({len(conversation.turns)}/{self.max_turns})"
            )
            return False

        # Check time limit
        if conversation.started_at:
            elapsed_minutes = (
                conversation.completed_at or conversation.created_at
            ) - conversation.started_at
            elapsed_minutes = elapsed_minutes.total_seconds() / 60

            if elapsed_minutes > self.max_duration_minutes:
                self.decisions["time_limit_stops"] += 1
                logger.info(
                    f"Conversation {conversation.conversation_id} stopped: "
                    f"time limit reached ({elapsed_minutes:.1f}/{self.max_duration_minutes}m)"
                )
                return False

        # Check quality-based limits (if enabled)
        if self.enable_quality_limits and len(conversation.turns) >= 3:
            recent_turns = conversation.turns[-3:]
            avg_quality = self._estimate_turn_quality(recent_turns)

            if avg_quality < self.min_quality_threshold:
                self.decisions["quality_limit_stops"] += 1
                logger.info(
                    f"Conversation {conversation.conversation_id} stopped: "
                    f"quality threshold not met ({avg_quality:.2f} < {self.min_quality_threshold})"
                )
                return False

        # Continue conversation
        self.decisions["continue_decisions"] += 1
        return True

    def get_completion_reason(
        self, conversation: ConversationAggregate
    ) -> Optional[CompletionReason]:
        """
        Get the reason why conversation should be completed.

        Args:
            conversation: Current conversation state

        Returns:
            Completion reason if conversation should stop, None if should continue
        """
        # This mirrors the logic in should_continue but returns reasons
        if len(conversation.turns) >= self.max_turns:
            return CompletionReason.TURN_LIMIT_REACHED

        if conversation.started_at:
            elapsed_minutes = (
                conversation.completed_at or conversation.created_at
            ) - conversation.started_at
            elapsed_minutes = elapsed_minutes.total_seconds() / 60

            if elapsed_minutes > self.max_duration_minutes:
                return CompletionReason.TIMEOUT

        if self.enable_quality_limits and len(conversation.turns) >= 3:
            recent_turns = conversation.turns[-3:]
            avg_quality = self._estimate_turn_quality(recent_turns)

            if avg_quality < self.min_quality_threshold:
                return CompletionReason.AGENT_FAILURE  # Quality issues treated as agent failure

        return None

    def _estimate_turn_quality(self, turns: List[ConversationTurnDomain]) -> float:
        """Estimate average quality of recent turns (simplified heuristic)."""
        if not turns:
            return 0.5

        quality_scores = []
        for turn in turns:
            if turn.response and turn.status == TurnStatus.COMPLETED:
                # Simple quality heuristic based on response length and content
                response_length = len(turn.response)

                if response_length < 10:
                    quality_scores.append(0.2)
                elif response_length > 500:
                    quality_scores.append(0.7)
                else:
                    # Scale from 0.3 to 0.9 based on length
                    quality_scores.append(0.3 + (response_length / 500) * 0.6)
            else:
                quality_scores.append(0.1)  # Failed turns get low quality

        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.3

    def get_metrics(self) -> Dict[str, Any]:
        """Get policy decision metrics."""
        return self.decisions.copy()


class ConsensusCompletionDetector(ICompletionDetector):
    """
    Completion detector that identifies when agents reach consensus or task completion.

    Uses simple heuristics to detect conversation completion conditions:
    - Consensus (agents agreeing)
    - Task completion (goals achieved)
    - Exploration exhaustion (no new insights)
    """

    def __init__(
        self,
        enable_consensus_detection: bool = True,
        consensus_threshold: float = 0.7,
        min_turns_for_consensus: int = 3,
    ):
        """Initialize completion detector."""
        self.enable_consensus_detection = enable_consensus_detection
        self.consensus_threshold = consensus_threshold
        self.min_turns_for_consensus = min_turns_for_consensus

        # Detection metrics
        self.detections = {
            "consensus_detected": 0,
            "task_completion_detected": 0,
            "no_completion_detected": 0,
        }

        logger.info(
            f"CompletionDetector initialized - consensus: {enable_consensus_detection}, "
            f"threshold: {consensus_threshold}, min_turns: {min_turns_for_consensus}"
        )

    async def detect_completion(self, context: ConversationContext) -> Optional[CompletionReason]:
        """
        Detect if conversation should be completed based on content analysis.

        Args:
            context: Current conversation context

        Returns:
            Completion reason if detected, None otherwise
        """
        if not self.enable_consensus_detection:
            self.detections["no_completion_detected"] += 1
            return None

        if len(context.turn_history) < self.min_turns_for_consensus:
            self.detections["no_completion_detected"] += 1
            return None

        # Analyze recent turns for consensus indicators
        recent_turns = context.recent_turns
        if not recent_turns:
            self.detections["no_completion_detected"] += 1
            return None

        # Simple consensus detection based on keywords
        consensus_score = self._analyze_consensus(recent_turns)

        if consensus_score >= self.consensus_threshold:
            self.detections["consensus_detected"] += 1
            logger.info(
                f"Consensus detected in conversation {context.conversation_id} "
                f"(score: {consensus_score:.2f})"
            )
            return CompletionReason.CONSENSUS_REACHED

        # Check for task completion indicators
        if self._analyze_task_completion(recent_turns):
            self.detections["task_completion_detected"] += 1
            logger.info(f"Task completion detected in conversation {context.conversation_id}")
            return CompletionReason.TASK_COMPLETED

        self.detections["no_completion_detected"] += 1
        return None

    def _analyze_consensus(self, turns: List[ConversationTurnDomain]) -> float:
        """Analyze turns for consensus indicators."""
        if not turns:
            return 0.0

        consensus_keywords = [
            "agree",
            "consensus",
            "agreed",
            "exactly",
            "precisely",
            "correct",
            "right",
            "yes",
            "absolutely",
            "definitely",
            "conclusion",
            "decided",
            "settled",
            "resolved",
        ]

        total_responses = 0
        consensus_responses = 0

        for turn in turns:
            if turn.response and turn.status == TurnStatus.COMPLETED:
                total_responses += 1
                response_lower = turn.response.lower()

                # Check for consensus keywords
                keyword_matches = sum(
                    1 for keyword in consensus_keywords if keyword in response_lower
                )
                if keyword_matches > 0:
                    consensus_responses += 1

        return consensus_responses / total_responses if total_responses > 0 else 0.0

    def _analyze_task_completion(self, turns: List[ConversationTurnDomain]) -> bool:
        """Analyze turns for task completion indicators."""
        if not turns:
            return False

        completion_keywords = [
            "completed",
            "finished",
            "done",
            "accomplished",
            "achieved",
            "solved",
            "resolved",
            "concluded",
            "final",
            "summary",
        ]

        # Check last 2 turns for completion indicators
        recent_turns = turns[-2:] if len(turns) >= 2 else turns

        for turn in recent_turns:
            if turn.response and turn.status == TurnStatus.COMPLETED:
                response_lower = turn.response.lower()

                # Check for completion keywords
                keyword_matches = sum(
                    1 for keyword in completion_keywords if keyword in response_lower
                )
                if keyword_matches >= 2:  # Require multiple completion indicators
                    return True

        return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get completion detection metrics."""
        return self.detections.copy()


# Factory functions for dependency injection


def create_turn_controller(
    response_generator: IAgentResponseGenerator,
) -> ConversationTurnController:
    """Factory function to create configured turn controller."""
    event_publisher = ConversationEventPublisher(enable_detailed_logging=True)
    return ConversationTurnController(
        response_generator=response_generator,
        event_publisher=event_publisher,
    )


def create_event_publisher() -> ConversationEventPublisher:
    """Factory function to create configured event publisher."""
    return ConversationEventPublisher(enable_detailed_logging=True)


def create_turn_limit_policy(
    max_turns: int = 10,
    max_duration_minutes: int = 30,
    enable_quality_limits: bool = False,
) -> TurnLimitPolicy:
    """Factory function to create configured turn limit policy."""
    return TurnLimitPolicy(
        max_turns=max_turns,
        max_duration_minutes=max_duration_minutes,
        enable_quality_limits=enable_quality_limits,
    )


def create_completion_detector(enable_consensus: bool = True) -> ConsensusCompletionDetector:
    """Factory function to create configured completion detector."""
    return ConsensusCompletionDetector(
        enable_consensus_detection=enable_consensus,
        consensus_threshold=0.7,
        min_turns_for_consensus=3,
    )

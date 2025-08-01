"""
Multi-Turn Agent Conversation Orchestrator Service

This module implements the core conversation orchestration service that manages
turn-based agent interactions following the architecture designed by the Nemesis Committee.

The orchestrator uses dependency injection, clean interfaces, and rich domain objects
to provide sophisticated conversation management capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from api.v1.models.agent_conversation import (
    AgentRole,
    CompletionReason,
    ConversationAggregate,
    ConversationConfig,
    ConversationEvent,
    ConversationStatus,
    ConversationTurnDomain,
    IAgentResponseGenerator,
    ICompletionDetector,
    IConversationEventPublisher,
    IConversationRepository,
    ITurnController,
    ITurnLimitPolicy,
)

logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    """
    Core conversation orchestrator that manages multi-turn agent conversations.

    This orchestrator implements the business logic for conversation lifecycle management,
    turn coordination, and completion detection as specified in Task 41.1.
    """

    def __init__(
        self,
        repository: IConversationRepository,
        response_generator: IAgentResponseGenerator,
        turn_controller: ITurnController,
        event_publisher: IConversationEventPublisher,
        turn_limit_policy: ITurnLimitPolicy,
        completion_detector: Optional[ICompletionDetector] = None,
    ):
        """Initialize the conversation orchestrator with dependencies."""
        self.repository = repository
        self.response_generator = response_generator
        self.turn_controller = turn_controller
        self.event_publisher = event_publisher
        self.turn_limit_policy = turn_limit_policy
        self.completion_detector = completion_detector

        # Active conversation tracking for lifecycle management
        self._active_conversations: Dict[str, ConversationAggregate] = {}
        self._conversation_tasks: Dict[str, asyncio.Task] = {}

        logger.info("ConversationOrchestrator initialized with dependencies")

    async def create_conversation(
        self,
        user_id: str,
        topic: str,
        participants: List[AgentRole],
        config: Optional[ConversationConfig] = None,
    ) -> ConversationAggregate:
        """
        Create a new conversation with specified participants.

        Args:
            user_id: ID of user creating the conversation
            topic: Conversation topic/prompt
            participants: List of agent roles participating
            config: Optional conversation configuration

        Returns:
            Created conversation aggregate

        Raises:
            ValueError: If participants list is invalid
        """
        if not participants:
            raise ValueError("Conversation must have at least one participant")

        if len(participants) > 10:
            raise ValueError("Conversation cannot have more than 10 participants")

        # Assign turn order if not specified
        for i, participant in enumerate(participants):
            if participant.turn_order == 0:
                participant.turn_order = i + 1

        # Create conversation aggregate
        config = config or ConversationConfig()
        conversation = ConversationAggregate(
            user_id=user_id,
            topic=topic,
            participants=participants,
            config=config,
        )

        # Persist conversation
        await self.repository.save_conversation(conversation)

        # Track active conversation
        self._active_conversations[conversation.conversation_id] = conversation

        # Publish creation event
        await self._publish_event(
            conversation,
            "conversation_created",
            {
                "user_id": user_id,
                "topic": topic,
                "participant_count": len(participants),
                "max_turns": config.max_turns,
            },
        )

        logger.info(
            f"Created conversation {conversation.conversation_id} with {len(participants)} participants"
        )
        return conversation

    async def start_conversation(self, conversation_id: str) -> ConversationAggregate:
        """
        Start an existing conversation.

        Args:
            conversation_id: ID of conversation to start

        Returns:
            Started conversation aggregate

        Raises:
            ValueError: If conversation cannot be started
        """
        conversation = await self._get_conversation(conversation_id)

        # Use domain aggregate business rule to start
        conversation.start_conversation()

        # Save state change
        await self.repository.save_conversation(conversation)

        # Start conversation execution task
        task = asyncio.create_task(self._execute_conversation(conversation))
        self._conversation_tasks[conversation_id] = task

        # Publish start event
        await self._publish_event(
            conversation,
            "conversation_started",
            {
                "participant_count": len(conversation.participants),
            },
        )

        logger.info(f"Started conversation {conversation_id}")
        return conversation

    async def pause_conversation(self, conversation_id: str) -> ConversationAggregate:
        """
        Pause an active conversation.

        Args:
            conversation_id: ID of conversation to pause

        Returns:
            Paused conversation aggregate
        """
        conversation = await self._get_conversation(conversation_id)

        # Use domain aggregate business rule to pause
        conversation.pause_conversation()

        # Cancel execution task
        if conversation_id in self._conversation_tasks:
            self._conversation_tasks[conversation_id].cancel()
            del self._conversation_tasks[conversation_id]

        # Save state change
        await self.repository.save_conversation(conversation)

        # Publish pause event
        await self._publish_event(
            conversation,
            "conversation_paused",
            {
                "turn_number": conversation.current_turn_number,
            },
        )

        logger.info(f"Paused conversation {conversation_id}")
        return conversation

    async def resume_conversation(self, conversation_id: str) -> ConversationAggregate:
        """
        Resume a paused conversation.

        Args:
            conversation_id: ID of conversation to resume

        Returns:
            Resumed conversation aggregate
        """
        conversation = await self._get_conversation(conversation_id)

        # Use domain aggregate business rule to resume
        conversation.resume_conversation()

        # Save state change
        await self.repository.save_conversation(conversation)

        # Restart conversation execution task
        task = asyncio.create_task(self._execute_conversation(conversation))
        self._conversation_tasks[conversation_id] = task

        # Publish resume event
        await self._publish_event(
            conversation,
            "conversation_resumed",
            {
                "turn_number": conversation.current_turn_number,
            },
        )

        logger.info(f"Resumed conversation {conversation_id}")
        return conversation

    async def stop_conversation(
        self, conversation_id: str, reason: CompletionReason = CompletionReason.MANUAL_STOP
    ) -> ConversationAggregate:
        """
        Stop a conversation manually.

        Args:
            conversation_id: ID of conversation to stop
            reason: Reason for stopping

        Returns:
            Stopped conversation aggregate
        """
        conversation = await self._get_conversation(conversation_id)

        # Complete conversation with specified reason
        conversation.complete_conversation(reason)

        # Cancel execution task
        if conversation_id in self._conversation_tasks:
            self._conversation_tasks[conversation_id].cancel()
            del self._conversation_tasks[conversation_id]

        # Remove from active tracking
        if conversation_id in self._active_conversations:
            del self._active_conversations[conversation_id]

        # Save final state
        await self.repository.save_conversation(conversation)

        # Publish stop event
        await self._publish_event(
            conversation,
            "conversation_stopped",
            {
                "reason": reason.value,
                "total_turns": len(conversation.turns),
            },
        )

        logger.info(f"Stopped conversation {conversation_id} with reason: {reason.value}")
        return conversation

    async def get_conversation(self, conversation_id: str) -> Optional[ConversationAggregate]:
        """Get conversation by ID."""
        return await self._get_conversation(conversation_id)

    async def list_active_conversations(self, user_id: str) -> List[ConversationAggregate]:
        """List active conversations for a user."""
        return await self.repository.list_active_conversations(user_id)

    async def _execute_conversation(self, conversation: ConversationAggregate) -> None:
        """
        Execute the main conversation loop.

        This method runs asynchronously to manage the conversation turns,
        handling timeouts, errors, and completion detection.
        """
        conversation_id = conversation.conversation_id
        logger.info(f"Starting execution loop for conversation {conversation_id}")

        try:
            while conversation.is_active:
                # Check if conversation should continue based on policy
                if not conversation.should_continue(self.turn_limit_policy):
                    completion_reason = self.turn_limit_policy.get_completion_reason(conversation)
                    conversation.complete_conversation(
                        completion_reason or CompletionReason.TURN_LIMIT_REACHED
                    )
                    break

                # Check for conversation timeout
                if self._is_conversation_timeout(conversation):
                    conversation.complete_conversation(CompletionReason.TIMEOUT)
                    break

                # Get next agent for turn
                next_agent = conversation.next_agent
                if not next_agent:
                    logger.error(f"No next agent found for conversation {conversation_id}")
                    conversation.complete_conversation(
                        CompletionReason.AGENT_FAILURE, "No agents available for next turn"
                    )
                    break

                # Execute the next turn
                try:
                    await self._execute_turn(conversation, next_agent)
                except Exception as e:
                    logger.error(
                        f"Turn execution failed for conversation {conversation_id}: {e}",
                        exc_info=True,
                    )
                    conversation.complete_conversation(
                        CompletionReason.AGENT_FAILURE, f"Turn execution failed: {str(e)}"
                    )
                    break

                # Check for automatic completion (consensus, task completion, etc.)
                if self.completion_detector:
                    completion_reason = await self.completion_detector.detect_completion(
                        conversation.current_context
                    )
                    if completion_reason:
                        conversation.complete_conversation(completion_reason)
                        break

                # Brief pause between turns to avoid overwhelming the system
                await asyncio.sleep(0.5)

                # Refresh conversation state in case of external modifications
                conversation = await self._get_conversation(conversation_id)
                if not conversation:
                    logger.warning(f"Conversation {conversation_id} was deleted during execution")
                    break

        except asyncio.CancelledError:
            logger.info(f"Conversation {conversation_id} execution was cancelled")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error in conversation {conversation_id} execution: {e}", exc_info=True
            )
            conversation.complete_conversation(
                CompletionReason.AGENT_FAILURE, f"Unexpected error: {str(e)}"
            )
        finally:
            # Clean up
            if conversation_id in self._active_conversations:
                del self._active_conversations[conversation_id]
            if conversation_id in self._conversation_tasks:
                del self._conversation_tasks[conversation_id]

            # Save final state
            await self.repository.save_conversation(conversation)

            # Publish completion event
            await self._publish_event(
                conversation,
                "conversation_execution_ended",
                {
                    "final_status": conversation.status.value,
                    "total_turns": len(conversation.turns),
                    "duration_minutes": conversation.duration_minutes,
                },
            )

            logger.info(
                f"Conversation {conversation_id} execution ended with status: {conversation.status.value}"
            )

    async def _execute_turn(self, conversation: ConversationAggregate, agent: AgentRole) -> None:
        """
        Execute a single conversation turn.

        Args:
            conversation: Current conversation state
            agent: Agent taking this turn
        """
        # Create turn domain object
        turn = ConversationTurnDomain(
            turn_number=conversation.current_turn_number,
            agent_id=agent.agent_id,
            agent_name=agent.name,
            prompt=conversation.current_context.create_agent_prompt(agent),
        )

        # Mark turn as started
        turn.mark_started()

        # Publish turn start event
        await self._publish_event(
            conversation,
            "turn_started",
            {
                "turn_number": turn.turn_number,
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
            },
        )

        try:
            # Execute turn through turn controller with timeout
            completed_turn = await asyncio.wait_for(
                self.turn_controller.execute_turn(turn, agent, conversation.current_context),
                timeout=conversation.config.turn_timeout_seconds,
            )

            # Validate response
            if not completed_turn.response:
                completed_turn.mark_failed("Empty response from agent")
            elif len(completed_turn.response) < conversation.config.min_response_length:
                completed_turn.mark_failed(
                    f"Response too short (minimum {conversation.config.min_response_length} characters)"
                )
            elif len(completed_turn.response) > conversation.config.max_response_length:
                completed_turn.mark_failed(
                    f"Response too long (maximum {conversation.config.max_response_length} characters)"
                )

            # Add completed turn to conversation
            conversation.add_turn(completed_turn)

            # Save conversation state
            await self.repository.save_conversation(conversation)

            # Publish turn completion event
            await self._publish_event(
                conversation,
                "turn_completed",
                {
                    "turn_number": completed_turn.turn_number,
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "status": completed_turn.status.value,
                    "duration_seconds": completed_turn.duration_seconds,
                    "response_length": len(completed_turn.response or ""),
                },
            )

            logger.info(
                f"Completed turn {completed_turn.turn_number} by {agent.name} in conversation {conversation.conversation_id}"
            )

        except asyncio.TimeoutError:
            turn.mark_timeout()
            conversation.add_turn(turn)
            await self.repository.save_conversation(conversation)

            await self._publish_event(
                conversation,
                "turn_timeout",
                {
                    "turn_number": turn.turn_number,
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "timeout_seconds": conversation.config.turn_timeout_seconds,
                },
            )

            logger.warning(
                f"Turn {turn.turn_number} by {agent.name} timed out in conversation {conversation.conversation_id}"
            )
            raise

        except Exception as e:
            turn.mark_failed(str(e))
            conversation.add_turn(turn)
            await self.repository.save_conversation(conversation)

            await self._publish_event(
                conversation,
                "turn_failed",
                {
                    "turn_number": turn.turn_number,
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "error": str(e),
                },
            )

            logger.error(
                f"Turn {turn.turn_number} by {agent.name} failed in conversation {conversation.conversation_id}: {e}"
            )
            raise

    def _is_conversation_timeout(self, conversation: ConversationAggregate) -> bool:
        """Check if conversation has exceeded its timeout."""
        if not conversation.started_at:
            return False

        elapsed_minutes = (datetime.now() - conversation.started_at).total_seconds() / 60
        return elapsed_minutes > conversation.config.conversation_timeout_minutes

    async def _get_conversation(self, conversation_id: str) -> ConversationAggregate:
        """Get conversation from cache or repository."""
        # Try active conversations first
        if conversation_id in self._active_conversations:
            return self._active_conversations[conversation_id]

        # Load from repository
        conversation = await self.repository.get_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation not found: {conversation_id}")

        # Cache if active
        if conversation.is_active:
            self._active_conversations[conversation_id] = conversation

        return conversation

    async def _publish_event(
        self, conversation: ConversationAggregate, event_type: str, data: Dict
    ) -> None:
        """Publish a conversation event."""
        event = ConversationEvent(
            conversation_id=conversation.conversation_id,
            event_type=event_type,
            data=data,
        )

        try:
            await self.event_publisher.publish_event(event)
        except Exception as e:
            logger.warning(
                f"Failed to publish event {event_type} for conversation {conversation.conversation_id}: {e}"
            )

    async def cleanup_completed_conversations(self, max_age_hours: int = 24) -> int:
        """
        Clean up old completed conversations from active tracking.

        Args:
            max_age_hours: Maximum age in hours for completed conversations

        Returns:
            Number of conversations cleaned up
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0

        # Find completed conversations older than cutoff
        to_remove = []
        for conv_id, conversation in self._active_conversations.items():
            if (
                conversation.is_completed
                and conversation.completed_at
                and conversation.completed_at < cutoff_time
            ):
                to_remove.append(conv_id)

        # Remove from active tracking
        for conv_id in to_remove:
            del self._active_conversations[conv_id]
            if conv_id in self._conversation_tasks:
                self._conversation_tasks[conv_id].cancel()
                del self._conversation_tasks[conv_id]
            cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} completed conversations")

        return cleaned_count

    def get_orchestrator_status(self) -> Dict:
        """Get current orchestrator status for monitoring."""
        active_count = len([c for c in self._active_conversations.values() if c.is_active])
        paused_count = len(
            [
                c
                for c in self._active_conversations.values()
                if c.status == ConversationStatus.PAUSED
            ]
        )
        completed_count = len([c for c in self._active_conversations.values() if c.is_completed])

        return {
            "total_conversations": len(self._active_conversations),
            "active_conversations": active_count,
            "paused_conversations": paused_count,
            "completed_conversations": completed_count,
            "running_tasks": len(self._conversation_tasks),
            "timestamp": datetime.now().isoformat(),
        }
